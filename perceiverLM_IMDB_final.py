from perceiver_io.perceiver_lm import PerceiverLM
from perceiver_io.perceiver_in import PerceiverIN

import os, sys
import torch
import torch.nn as nn
import transformers

from deepmind_assets import bytes_tokenizer
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import scipy.sparse as sp
# import xclib.evaluation.xc_metrics as xc_metrics
# from utils import csr_to_pad_tensor, ToD, read_sparse_mat, XCMetrics, _c
from utils import csr_to_pad_tensor, ToD, _c
from torch.nn.utils.rnn import pad_sequence
import copy
import random

# The tokenizer is just UTF-8 encoding (with an offset)
tokenizer = bytes_tokenizer.BytesTokenizer()


command = "--dataset IMDB"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--project', default='PerceiverIO_pre-sattn_layernorm_decod_classf')
parser.add_argument('--dataset', default='IMDB')
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args(command.split())


args.expname = args.project
args.maxlen = 2048
args.vocab_size = 262
args.embed_dim = 768
args.num_latents = 256

args.n_epochs = 15
args.xc_lr = 1e-3
args.enc_lr = 1e-4
args.bsz = 4
args.dropout = 0.5
args.warmup = 0.1
args.loss_with_logits = True
args.amp = True
args.eval_interval = 2
args.numy = 2

args.per_label_task = False
args.per_token_decoder = False

# args.mode = 'all'
# args.mode = 'decoder+classifier'
# args.mode = 'pre_san+decoder+classifier'
args.mode = 'pre_san+san_layer_norm+decoder+classifier'

OUT_DIR = f'{args.project}/{args.dataset}'
os.makedirs(OUT_DIR, exist_ok=True)


encoder = PerceiverLM(vocab_size=args.vocab_size, 
                      max_seq_len=args.maxlen, 
                      embedding_dim=args.embed_dim, 
                      num_latents=args.num_latents, 
                      latent_dim=1280, 
                      qk_out_dim=256, 
                      dropout=0,
                      num_self_attn_per_block=26, 
                      per_token_decoder=args.per_token_decoder, 
                      num_query_tasks=args.numy if args.per_label_task else 1)

encoder.load_pretrained("deepmind_assets/language_perceiver_io_bytes.pickle")


from torchtext.datasets import IMDB
trainset = IMDB(root='./data', split='train')
testset = IMDB(root='./data', split='test')

train_set = [elem for elem in trainset]
test_set = [elem for elem in testset]
random.shuffle(train_set)
random.shuffle(test_set)


trn_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.bsz,
    num_workers=1,
    # collate_fn=XMLCollator(trn_dataset),
    pin_memory=True)

tst_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.bsz,
    num_workers=1,
    # collate_fn=XMLCollator(tst_dataset),
    pin_memory=True)


class Net(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.numy = args.numy
        self.dropout = nn.Dropout(args.dropout)
        if args.per_label_task:
            self.w = nn.Sequential(nn.Linear(args.embed_dim, 2*args.embed_dim), 
                                   nn.ReLU(), 
                                   nn.Linear(2*args.embed_dim, 1))
        else:
            self.w = nn.Linear(args.embed_dim, args.numy)
        
    def get_device(self):
        return list(self.parameters())[0].device
    
    def forward(self, inputs, mask):
        # mask = b['xfts']['input_mask']
        embs = self.encoder(inputs, mask)
        
        if self.encoder.per_token_decoder:
            embs = embs * mask.unsqueeze(-1) / mask.sum(dim=-1).reshape(-1, 1, 1)
            embs = embs.sum(dim=1)
        else:
            embs = embs.squeeze()
        out = self.w(self.dropout(embs))
        return out.squeeze()
    
    def predict(self, tst_loader, K=100):
        tst_X_Y = tst_loader.dataset.labels
        data = np.zeros((tst_X_Y.shape[0], K))
        inds = np.zeros((tst_X_Y.shape[0], K)).astype(np.int32)
        indptr = np.arange(0, tst_X_Y.shape[0]*K+1, K)
        self.eval()

        with torch.no_grad():
            for b in tqdm(tst_loader, leave=True, desc='Evaluating'):
                b = ToD(b, self.get_device())
                out = self(b)
                top_data, top_inds = torch.topk(out, K)
                data[b['ids'].cpu()] = top_data.detach().cpu().numpy()
                inds[b['ids'].cpu()] = top_inds.detach().cpu().numpy()
                del top_data, top_inds, b, out

        torch.cuda.empty_cache()
        score_mat = sp.csr_matrix((data.ravel(), inds.ravel(), indptr), tst_X_Y.shape)
        
        return score_mat
    
class OvABCELoss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(OvABCELoss, self).__init__()
        if args.loss_with_logits:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, model, b):
        out = model(b)
        targets = torch.zeros((out.shape[0], out.shape[1]+1), device=out.device).scatter_(1, b['y']['inds'], 1)[:, :-1]
        loss = self.criterion(out, targets)
        return loss


net = Net(encoder, args)
criterion = nn.CrossEntropyLoss()


optim_wrap = {
    'xc' : {'class': torch.optim.Adam, 'params': [], 'args': {'lr': args.xc_lr}},
    'enc': {'class': transformers.optimization.AdamW, 'params': [], 
            'args': {'lr': args.enc_lr, 'eps': 1e-06, 'weight_decay': 0.01}}
    }

for n,p in net.named_parameters():
    if 'query_task_embedding' in n or p.shape[-1] == args.numy or p.shape[0] == args.numy: 
        optim_wrap['xc']['params'].append((n, p))
    else: 
        optim_wrap['enc']['params'].append((n, p))
        
optims = []
for k, v in optim_wrap.items():
    if len(v['params']) > 0: optims.append(v['class']([x[1] for x in v['params']], **v['args']))
        

total_steps = len(trn_loader)*args.n_epochs
schedulers = [transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps=int(args.warmup*total_steps), num_training_steps=total_steps) for optim in optims]


# Sets requires_grad=False for some params and Prints all parameter names which have requires_grad=True
for n, p in net.named_parameters():
    if n.startswith('encoder.'):
        if args.mode == 'decoder+classifier':
            if not n.startswith('encoder.query') and not n.startswith('encoder.perceiver.decoder'):
                p.requires_grad = False
                continue
        if args.mode == 'pre_san+decoder+classifier':
            if n.startswith('encoder.perceiver.encoder.self_attention'):
                p.requires_grad = False
                continue
        if args.mode == 'pre_san+san_layer_norm+decoder+classifier':
            if n.startswith('encoder.perceiver.encoder.self_attention') and not 'layer_norm' in n:
                p.requires_grad = False
                continue
    print(n)



net.to(args.device);


def pad(max_sequence_length: int, inputs, input_mask):
    input_len = inputs.shape[1]
    if input_len > max_sequence_length:
      return inputs[0,:max_sequence_length][None], input_mask[0,:max_sequence_length][None]
    pad_len = max_sequence_length - input_len
    padded_inputs = np.pad(
      inputs,
      pad_width=((0, 0), (0, pad_len)),
      constant_values=tokenizer.pad_token)
    padded_mask = np.pad(
      input_mask,
      pad_width=((0, 0), (0, pad_len)),
      constant_values=0)
    return padded_inputs, padded_mask



scaler = torch.cuda.amp.GradScaler()
train_loss = []
valid_accuracy = []
best_valid_acc = 0
best_epoch = 0
for epoch in range(args.n_epochs):
    print(epoch)
    net.train()
    cum_loss = 0; ctr = 0
    # t = tqdm(trn_loader, desc='Epoch: 0, Loss: 0.0', leave=True)
          
    for y, X in tqdm(trn_loader):        
        for optim in optims: optim.zero_grad()
        # b = ToD(b, args.device)
        input_tokens = []
        input_mask = []
        for input_str in X:
          tokens = tokenizer.to_int(input_str)[None]
          mask = np.ones_like(tokens)

          inputs, mask = pad(args.maxlen, tokens, mask)
          input_tokens.append(inputs[0,:])
          input_mask.append(mask[0,:])

        input_tokens = np.array(input_tokens)
        input_mask = np.array(input_mask)

        out = net.forward(torch.tensor(input_tokens).to(args.device), torch.tensor(input_mask).to(args.device))

        y = torch.tensor([0 if elem == 'neg' else 1 for elem in y])
        # del input_tokens, input_mask
        # torch.cuda.empty_cache()
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = criterion(out, y.to(args.device))
            
        if args.amp:
            scaler.scale(loss).backward()
            for optim in optims: scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            for optim in optims: optim.step()
                
        for sch in schedulers: sch.step()
        cum_loss += loss.item()
        ctr += 1
        # del out
        # torch.cuda.empty_cache()
        if ctr%1000 == 0:
            print('Epoch: %d/%d, Loss: %.4E'%(epoch, args.n_epochs, (cum_loss/ctr)))
            train_loss.append(cum_loss/ctr)
        # t.set_description('Epoch: %d/%d, Loss: %.4E'%(epoch, args.n_epochs, (cum_loss/ctr)), refresh=True)
    print(f'mean loss after epoch {epoch}/{args.n_epochs}: {"%.4E"%(cum_loss/ctr)}', flush=True)
    if epoch%args.eval_interval == 0 or epoch == (args.n_epochs-1):
        print("Testing on validation data")
        corrects = 0
        net.eval()
        # t = tqdm(tst_loader, leave=True)
        for y, X in tqdm(tst_loader):        
            input_tokens = []
            input_mask = []
            for input_str in X:
              tokens = tokenizer.to_int(input_str)[None]
              mask = np.ones_like(tokens)

              inputs, mask = pad(args.maxlen, tokens, mask)
              input_tokens.append(inputs[0,:])
              input_mask.append(mask[0,:])

            input_tokens = np.array(input_tokens)
            input_mask = np.array(input_mask)

            out = net.forward(torch.tensor(input_tokens).to(args.device), torch.tensor(input_mask).to(args.device))
            # del input_mask, input_tokens
            # torch.cuda.empty_cache()
            y = torch.tensor([0 if elem == 'neg' else 1 for elem in y])
            corrects += int(sum(out.argmax(dim=1) == y.to(args.device)).cpu().detach().item())
            # del out
            # torch.cuda.empty_cache()
        valid_accuracy.append(corrects/args.bsz/len(tst_loader))
        print(valid_accuracy[-1])

        if valid_accuracy[-1] > best_valid_acc:
            best_valid_acc = valid_accuracy[-1]
            print(f'Found new best model with Accuarcy: {"%.5f"%best_valid_acc}\n')
            torch.save(net.state_dict(), f'{OUT_DIR}/model.pt')
            best_epoch = epoch
    sys.stdout.flush()
    np.save(f'{OUT_DIR}/train_loss.npy', np.array(train_loss))
    np.save(f'{OUT_DIR}/valid_acc.npy', np.array(valid_accuracy))

print("Best epoch", best_epoch)





