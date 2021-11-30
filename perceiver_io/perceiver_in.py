from typing import Optional
import torch
from torch import nn
import pickle
from perceiver_io import PerceiverEncoder, PerceiverDecoder, PerceiverIO

class PerceiverIN(nn.Module):
    """Image net PerceiverIO."""
    def __init__(
        self,
        embedding_dim: int = 322,
        num_latents: int = 512,
        latent_dim: int = 1024,
        qk_out_dim = 322,
        v_out_dim = 322,
        num_self_attn_heads=8,
        num_cross_attn_heads=1,
        num_decoder_attn_heads=1,
        self_attn_widening_factor=1,
        cross_attn_widening_factor=1,
        num_blocks=8,
        num_self_attn_per_block=6,
        dropout: float = 0.0,
        per_token_decoder = True,
        num_query_tasks = 1
    ):
        super().__init__()
        self.per_token_decoder = per_token_decoder
        self.num_query_tasks = num_query_tasks        
        self.query_task_embedding = nn.Embedding(num_query_tasks, latent_dim)
        
        if v_out_dim is None: v_out_dim = latent_dim
        encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=embedding_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            self_qk_out_dim=None,
            self_v_out_dim=None,
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
            query_dim=latent_dim,
            qk_out_dim=None,
            v_out_dim=None,
            num_heads=num_decoder_attn_heads,
            widening_factor=cross_attn_widening_factor,
            projection_dim=None,
            use_query_residual=True
        )
        self.perceiver = PerceiverIO(encoder, decoder)
    
    def load_pretrained(self, fname, num_layers=6):
        params = None
        with open(fname, "rb") as f:
            params = pickle.loads(f.read())['params']
        state_dict = {}
        model_enc_base = 'perceiver.encoder.'
        params_enc_base = 'perceiver_encoder/~/'

        state_dict['query_task_embedding.weight'] = params['classification_decoder/~/basic_decoder/~/trainable_position_encoding']['pos_embs']
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
        state_dict = copy_attention_params(state_dict, f'perceiver.decoder.cross_attention.', f'classification_decoder/~/basic_decoder/cross_attention/')

        for i in range(6):
            state_dict = copy_attention_params(state_dict, f'{model_enc_base}self_attention_block.{i}.', f'{params_enc_base}self_attention{"_%d"%i if i else ""}/')

        state_dict = {k: torch.tensor(v) for k,v in state_dict.items()}
        print(self.load_state_dict(state_dict, strict=False))