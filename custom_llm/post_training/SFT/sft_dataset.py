import os
import numpy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .sft_utils import (format_input,
                        custom_collate_fn)


num_workers = 0
batch_size = 2

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, device="cpu"):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry["output"]}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        encoded_text = self.encoded_texts[index]
        # Return input and target tensors with correct shape and device
        inputs = torch.tensor(encoded_text[:-1], device=self.device).unsqueeze(0)
        targets = torch.tensor(encoded_text[1:], device=self.device).unsqueeze(0)
        return inputs, targets
    
    def __len__(self):
        return len(self.data)
    


    

class SFTDataloaders:
    def __init__(self, train_data, val_data, test_data, tokenizer, device="cpu"):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.device = device

    def data_loaders(self):
        train_dataset = InstructionDataset(self.train_data, self.tokenizer, device=self.device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )

        val_dataset = InstructionDataset(self.val_data, self.tokenizer, device=self.device)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        test_dataset = InstructionDataset(self.test_data, self.tokenizer, device=self.device)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        return train_dataset, val_dataloader, test_dataloader