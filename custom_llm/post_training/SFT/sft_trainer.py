import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sft_utils import format_input
from pretrain.train import train_model


class FineTuneModel:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 device,
                 num_epochs,
                 tokenizer,
                 val_data,
                 eval_freqs=5,
                 eval_iter=5,
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.eval_freqs = eval_freqs
        self.eval_iter = eval_iter
        self.val_data =val_data

    def train(self):
        start_time = time.time()
        train_losses, val_losses, token_seen = train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=self.num_epochs,
            eval_freqs=self.eval_freqs,
            eval_iter=self.eval_iter,
            start_context=format_input(self.val_data[0]),
            tokenizer=self.tokenizer
        )

        end_time = time.time()
        execution_time_in_minutes = (end_time - start_time) / 60
        print(f"training completed in {execution_time_in_minutes:.2f} minutes.")
        return self.model
        
