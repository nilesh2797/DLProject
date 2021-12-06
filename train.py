from perceiver_io.perceiver_lm import PerceiverLM
from perceiver_io.perceiver_in import PerceiverIN

import os, sys
import torch
import torch.nn as nn
import transformers
import torchvision
import torchvision.transforms as transforms
import pickle
with open("deepmind_assets/language_perceiver_io_bytes.pickle", "rb") as f:
    params = pickle.loads(f.read())
from deepmind_assets import bytes_tokenizer
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import scipy.sparse as sp
import xclib.evaluation.xc_metrics as xc_metrics
from utils import csr_to_pad_tensor, ToD, read_sparse_mat, XCMetrics, _c
from torch.nn.utils.rnn import pad_sequence

# The tokenizer is just UTF-8 encoding (with an offset)
tokenizer = bytes_tokenizer.BytesTokenizer()
command = "--dataset cifar10"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--project', default='PerceiverIO')
parser.add_argument('--dataset', default='EURLex-4K')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args(command.split())

args.expname = f'{args.project}-image'
args.maxlen = 2048

args.n_epochs = 25
args.lr = 5e-4
args.bsz = 16
args.dropout = 0.5
args.warmup = 0.1
args.loss_with_logits = True
args.amp = False
args.eval_interval = 1

OUT_DIR = f'Results/{args.expname}/{args.dataset}/IN'
os.makedirs(OUT_DIR, exist_ok=True)

args.img_size = 32
if args.dataset == 'tiny-imagenet':
    args.img_size = 64
elif args.dataset == 'stl10':
    args.img_size = 96
elif args.dataset == 'cifar10':
    args.img_size = 32
    
transform_train = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomCrop(args.img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    args.numy = 10
elif args.dataset == 'tiny-imagenet':
    fix_tin_val_folder('./data/tiny-imagenet-200/val')
    trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_test)
    args.numy = 200
elif args.dataset == 'stl10':
    trainset = torchvision.datasets.STL10(root='./data', split='train', download=False, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data', split='test', download=False, transform=transform_test)
    args.numy = 10

args.per_label_task = False
args.per_token_decoder = False
args.num_latents = 512
args.latent_dim = 1024
args.embed_dim = 322

from perceiver_io.perceiver_in import PerceiverIN
encoder = PerceiverIN(num_blocks=2)
encoder.load_pretrained('deepmind_assets/imagenet_perceiver.pystate')

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

args.patch_height = 4
args.patch_width = 4
args.num_patches = (args.img_size // args.patch_height) * (args.img_size // args.patch_width)
args.patch_dim = 3 * args.patch_height * args.patch_width

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsz, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bsz, shuffle=False, num_workers=4)

class Net(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_height, p2 = args.patch_width),
            nn.Linear(args.patch_dim, args.embed_dim),
        )
        self.encoder = encoder
        self.position_embedding = self.encoder.position_embedding if hasattr(self.encoder, 'position_embedding') else nn.Embedding(args.num_patches, args.embed_dim)
        self.numy = args.numy
        self.dropout = nn.Dropout(args.dropout)
        if args.per_label_task:
            self.w = nn.Sequential(nn.Linear(args.embed_dim, 2*args.embed_dim), 
                                   nn.ReLU(), 
                                   nn.Linear(2*args.embed_dim, 1))
        else:
            self.w = nn.Linear(args.latent_dim, args.numy)
        
    def get_device(self):
        return list(self.parameters())[0].device
    
    def forward(self, b):
        patch_embs = self.to_patch_embedding(b)
        seq_len = patch_embs.size(1)
        batch_size = patch_embs.size(0)
        
        pos_ids = torch.arange(seq_len, device=patch_embs.device).view(1, -1)
        pos_embs = self.position_embedding(pos_ids)
        embs = patch_embs + pos_embs
        
        if args.per_token_decoder:
            query_embs = self.encoder.query_position_embedding(pos_ids).repeat(batch_size, 1, 1)
            query_mask = None
        else:
            query_embs = self.encoder.query_task_embedding.weight.repeat(batch_size, 1, 1)
            query_mask = None
            
        embs = self.encoder.perceiver(
            inputs=embs,
            query=query_embs,
            input_mask=None,
            query_mask=None
        )
        
        if self.encoder.per_token_decoder:
            embs = embs.mean(dim=1)
        else:
            embs = embs.squeeze()
    
        out = self.w(self.dropout(embs))
        return out.squeeze()

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

optims = [transformers.optimization.AdamW(net.parameters(), **{'lr': args.lr, 'eps': 1e-06, 'weight_decay': 0.01})]
total_steps = len(trainloader)*args.n_epochs
schedulers = [transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps=int(args.warmup*total_steps), num_training_steps=total_steps) for optim in optims]
net.to(args.device)
#net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
def evaluate(net, testloader, epoch=-1):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    t = tqdm(testloader, desc='', leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description(' '.join([str(batch_idx), str(len(testloader)), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)]))
    
    acc = 100.*correct/total
    loss = test_loss/(batch_idx+1)
    return loss, acc

scaler = torch.cuda.amp.GradScaler()
best_acc = -100
for epoch in range(args.n_epochs):
    net.train()
    cum_loss = 0; ctr = 0
    t = tqdm(trainloader, desc='Epoch: 0, Loss: 0.0', leave=True)
          
    for b in t:        
        for optim in optims: optim.zero_grad()
        b = ToD({'input': b[0], 'label': b[1]}, args.device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = net(b['input'])
        loss = criterion(out, b['label'])
        
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
        t.set_description('Epoch: %d/%d, Loss: %.4E'%(epoch, args.n_epochs, (cum_loss/ctr)), refresh=True)
    
    print(f'mean loss after epoch {epoch}/{args.n_epochs}: {"%.4E"%(cum_loss/ctr)}', flush=True)
    if epoch%args.eval_interval == 0 or epoch == (args.n_epochs-1):
        test_loss, test_acc = evaluate(net, testloader)

        if test_acc > best_acc:
            best_acc = test_acc
            print(f'Found new best model with acc: {"%.2f"%best_acc}\n')
            with open(f'{OUT_DIR}/log.txt', 'a') as f:
                print(f'epoch: {epoch}, test acc: {test_acc}, train loss: {cum_loss/ctr}, test loss: {test_loss}', file=f)
            torch.save(net.state_dict(), f'{OUT_DIR}/model.pt')
    sys.stdout.flush()

test_loss, test_acc = evaluate(net, trainloader)

