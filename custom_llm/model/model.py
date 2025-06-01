import torch
import torch.nn as nn
import torch.nn.functional as F





"""
    This file contains the implementation of the MultiHeadAttention, RMSNorm, GELU, Feedforward, TransformerBlock and LLMModel classes.
    this is the main model for the custom LLM.
"""



class MultiHeadAttention(nn.Module):
    """
    This class implements the MultiHeadAttention mechanism.
    It takes in the input dimensions, the output dimensions, the context length, the dropout rate, the number of heads, and the qkv bias.
    It then initializes the weights and biases for the query, key, value, and output projections.
    It also initializes the mask for the attention mechanism.
    """
    def __init__(self, d_in:int, d_out:int, context_length:int, dropout:float, num_heads:int, qkv_bias:bool=False) -> None:
        super().__init__()
        assert (d_out % num_heads == 0) , "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)

        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill(mask_bool, -torch.inf)

        atten_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vec = (atten_weights @ values).transpose(1,2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec)
        return context_vec

        


    
class RMSNorm(nn.Module):
    """
    This class implements the RMSNorm normalization.
    It takes in the dimension of the input, and the epsilon value.
    It then initializes the weight for the normalization.
    """
    def __init__(self, dim:int, eps:float=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float().type_as(x))
        return output * self.weight        




class GELU(nn.Module):
    """
    This class implements the GELU activation function.
    It takes in the input tensor and returns the output tensor.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*( x + 0.44715*torch.pow(x, 3))))




class Feedforward(nn.Module):
    """
    This class implements the Feedforward network.
    It takes in the configuration dictionary and returns the output tensor.
    """
    def __init__(self, cfg:dict) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])

        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    



class TransformerBlock(nn.Module):
    """
    This class implements the TransformerBlock.
    It takes in the configuration dictionary and returns the output tensor.
    """
    def __init__(self, cfg:dict) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = Feedforward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
    
    

class LLMModel(nn.Module):
    """
    This class implements the LLMModel.
    It takes in the configuration dictionary and returns the output tensor.
    """
    def __init__(self, cfg:dict) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        
        # Initialize weights for better memory efficiency
        self.apply(self._init_weights)
        
    def _init_weights(self, module:nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx:torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        
        # Use mixed precision for embeddings with updated syntax
        with torch.amp.autocast('cuda'):
            tok_embeds = self.tok_emb(in_idx)
            pos_emb = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
            )
            x = tok_embeds + pos_emb
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            
        return logits