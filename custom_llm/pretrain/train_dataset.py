import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


"""
    This file contains the implementation of the GPTDataset class.
    this is the main dataset for the custom LLM.
"""

class GPTDataset(Dataset):
    """
    This class implements the GPTDataset.
    It takes in the data, tokenizer, max length, and stride.
    It then initializes the input ids and target ids.
    """
    def __init__(self, data, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(data)

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    





    
