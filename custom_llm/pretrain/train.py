import os
import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from generate import generate_and_print_simple
from utils import (
    calc_loss_per_batch,
    evaluate_model
)


"""
    This file contains the implementation of the train_model function.
    this is the main training function for the custom LLM.
"""

def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freqs,
        eval_iter,
        start_context,
        tokenizer
):
    """
    This function trains the model.
    It takes in the model, train loader, val loader, optimizer, device, num epochs, eval freqs, eval iter, start context, and tokenizer.
    It then trains the model.
    """
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1 
    
    # Set up gradient accumulation
    accumulation_steps = 4  # Accumulate gradients for 4 steps
    optimizer.zero_grad()  # Initialize gradients

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)
        for i, (input_batch, target_batch) in enumerate(pbar):
            # Forward pass
            loss = calc_loss_per_batch(
                input_batch, target_batch, model, device
            )
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freqs == 0:
                # Clear cache before evaluation
                torch.cuda.empty_cache()
                
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.3f}',
                    'val_loss': f'{val_loss:.3f}'
                })

        # Clear cache after each epoch
        torch.cuda.empty_cache()
        
        generate_and_print_simple(model,
                              tokenizer,
                              device,
                              start_context
        )
    print("model training has been completed")
    return train_losses, val_losses, start_context


        
                

