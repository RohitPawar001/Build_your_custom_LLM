import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as f
from utils import (
    text_to_token_ids,
    token_ids_to_text
)


"""
    This file contains the implementation of the generate function.
    this is the main generate function for the custom LLM.
"""


def generate(model,
             idx,
             max_ne_token,
             context_size,
             temperature=0.0,
             top_k=None,
             eos_id=None):
    
    """
    This function generates the text for the custom LLM.
    It takes in the model, idx, max new tokens, context size, temperature, top k, and eos id.
    It then generates the text.
    """

    for _ in range(max_ne_token):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_simple(model,
                              tokenizer,
                              device,
                              start_context):
    
    """
    This function generates the text for the custom LLM.
    It takes in the model, tokenizer, device, and start context.
    It then generates the text.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_ne_token=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def generate_text_simple(model,
                         idx,
                         max_new_tokens,
                         context_size):
    
    """
    This function generates the text for the custom LLM.
    It takes in the model, idx, max new tokens, and context size.
    It then generates the text.
    """
    idx= idx.to("cuda")
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx



