from typing import Optional
import torch
from torch import nn
import pickle
from perceiver_io import PerceiverEncoder, PerceiverDecoder, PerceiverIO

class PerceiverLM(nn.Module):
    """Encoder-decoder based language model."""
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        num_latents: int = 256,
        latent_dim: int = 1280,
        qk_out_dim = 8*32,
        v_out_dim = None,
        num_self_attn_heads=8,
        num_cross_attn_heads=8,
        num_decoder_attn_heads=8,
        self_attn_widening_factor=1,
        cross_attn_widening_factor=1,
        num_blocks=1,
        num_self_attn_per_block=12,
        dropout: float = 0.0,
        per_token_decoder = True,
        num_query_tasks = 1
    ):
        super().__init__()
        self.per_token_decoder = per_token_decoder
        self.num_query_tasks = num_query_tasks
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.query_position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.decoder_token_bias = nn.Parameter(torch.randn(vocab_size))
        
        self.query_task_embedding = nn.Embedding(num_query_tasks, embedding_dim)
        
        if v_out_dim is None: v_out_dim = latent_dim
        encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            num_self_attn_per_block=num_self_attn_per_block,
            num_blocks=num_blocks,
            num_self_attn_heads=num_self_attn_heads,
            num_cross_attn_heads=num_cross_attn_heads,
            cross_attn_widening_factor=cross_attn_widening_factor,
            self_attn_widening_factor=self_attn_widening_factor,
            dropout=dropout,
        )
        decoder = PerceiverDecoder(
            latent_dim=latent_dim,
            query_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=embedding_dim,
            num_heads=num_decoder_attn_heads,
            widening_factor=cross_attn_widening_factor,
            projection_dim=None
        )
        self.perceiver = PerceiverIO(encoder, decoder)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tasks: Optional[torch.Tensor] = None,
        tasks_mask: Optional[torch.Tensor] = None,
        mlm = False,
    ):
        """
        Args:
            inputs: Tensor of token ids.
            mask: Token mask. Mask values selected in [0, 1]. Defaults to None.
        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size).
        """
        seq_len = inputs.size(1)
        batch_size = inputs.size(0)
        token_embeddings = self.token_embedding(inputs)
        positions_ids = torch.arange(seq_len, device=inputs.device).view(1, -1)
        position_embeddings = self.position_embedding(positions_ids)
        embeddings = token_embeddings + position_embeddings
        
        if self.per_token_decoder:
            query_embeddings = self.query_position_embedding(positions_ids).repeat(batch_size, 1, 1)
            query_mask = mask
        elif tasks is None:
            query_embeddings = self.query_task_embedding.weight.repeat(batch_size, 1, 1)
            query_mask = None
        else:
            query_embeddings = self.query_task_embedding(tasks)
            query_mask = tasks_mask
        outputs = self.perceiver(
            inputs=embeddings,
            query=query_embeddings,
            input_mask=mask,
            query_mask=query_mask
        )
        if mlm:
            logits = torch.matmul(outputs, self.token_embedding.weight.T) + self.decoder_token_bias
            return logits
        return outputs
    
    def load_pretrained(self, fname, num_layers=26):
        params = None
        with open(fname, "rb") as f:
            params = pickle.loads(f.read())
        state_dict = {}
        model_enc_base = 'perceiver.encoder.'
        params_enc_base = 'perceiver_encoder/~/'

        state_dict['token_embedding.weight'] = params['embed']['embeddings']
        state_dict['decoder_token_bias'] = params['embedding_decoder']['bias']
        state_dict['position_embedding.weight'] = params['trainable_position_encoding']['pos_embs']
        state_dict['query_position_embedding.weight'] = params['basic_decoder/~/trainable_position_encoding']['pos_embs']
        state_dict[f'{model_enc_base}latents'] = params[f'{params_enc_base}trainable_position_encoding']['pos_embs']

        def copy_attention_params(state_dict, model_base, params_base):
            state_dict[f'{model_base}attention.q.weight'] = params[f'{params_base}attention/linear']['w'].T
            state_dict[f'{model_base}attention.q.bias'] = params[f'{params_base}attention/linear']['b']
            state_dict[f'{model_base}attention.k.weight'] = params[f'{params_base}attention/linear_1']['w'].T
            state_dict[f'{model_base}attention.k.bias'] = params[f'{params_base}attention/linear_1']['b']
            state_dict[f'{model_base}attention.v.weight'] = params[f'{params_base}attention/linear_2']['w'].T
            state_dict[f'{model_base}attention.v.bias'] = params[f'{params_base}attention/linear_2']['b']
            state_dict[f'{model_base}attention.projection.weight'] = params[f'{params_base}attention/linear_3']['w'].T
            state_dict[f'{model_base}attention.projection.bias'] = params[f'{params_base}attention/linear_3']['b']

            if 'self_attention' in params_base:
                state_dict[f'{model_base}layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
                state_dict[f'{model_base}layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
                state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
                state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
            else:
                state_dict[f'{model_base}q_layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
                state_dict[f'{model_base}q_layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
                state_dict[f'{model_base}kv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
                state_dict[f'{model_base}kv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
                state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_2']['scale']
                state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_2']['offset']

            state_dict[f'{model_base}mlp.mlp.0.weight'] = params[f'{params_base}mlp/linear']['w'].T
            state_dict[f'{model_base}mlp.mlp.0.bias'] = params[f'{params_base}mlp/linear']['b']
            state_dict[f'{model_base}mlp.mlp.2.weight'] = params[f'{params_base}mlp/linear_1']['w'].T
            state_dict[f'{model_base}mlp.mlp.2.bias'] = params[f'{params_base}mlp/linear_1']['b']
            return state_dict

        state_dict = copy_attention_params(state_dict, f'{model_enc_base}cross_attn.', f'{params_enc_base}cross_attention/')
        state_dict = copy_attention_params(state_dict, f'perceiver.decoder.cross_attention.', f'basic_decoder/cross_attention/')

        for i in range(num_layers):
            state_dict = copy_attention_params(state_dict, f'{model_enc_base}self_attention_block.{i}.', f'{params_enc_base}self_attention{"_%d"%i if i else ""}/')

        state_dict = {k: torch.tensor(v) for k,v in state_dict.items()}
        print(self.load_state_dict(state_dict, strict=False))
