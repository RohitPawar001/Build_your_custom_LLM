import os
import urllib
import json
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately complates the request."
        f"\n\n## Instruction:\n{entry["instruction"]}"
    )
    input_text = (
        f"\n\n### Input:\n{entry["input"]}" if entry["input"] else " "

    )

    return instruction_text + input_text




def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    # Unzip the batch into inputs and targets
    input_batch, target_batch = zip(*batch)
    
    # Get the maximum length in the batch
    batch_max_length = max(item.shape[1] for item in input_batch)
    
    # Pad and stack inputs
    padded_inputs = []
    padded_targets = []
    
    for inputs, targets in zip(input_batch, target_batch):
        # Ensure tensors are on the correct device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Pad inputs
        pad_length = batch_max_length - inputs.shape[1]
        if pad_length > 0:
            pad_tensor = torch.full((1, pad_length), pad_token_id, device=device)
            inputs = torch.cat([inputs, pad_tensor], dim=1)
            pad_target = torch.full((1, pad_length), ignore_index, device=device)
            targets = torch.cat([targets, pad_target], dim=1)
        
        if allowed_max_length is not None:
            inputs = inputs[:, :allowed_max_length]
            targets = targets[:, :allowed_max_length]
            
        padded_inputs.append(inputs)
        padded_targets.append(targets)
    
    # Stack the tensors
    input_tensors = torch.cat(padded_inputs, dim=0)
    target_tensors = torch.cat(padded_targets, dim=0)
    
    return input_tensors, target_tensors






def save_fine_tunned_model(model, optimizer):
    if os.path.exists(os.path.join(os.getcwd(),"fine_tunned_model_and_optimizer.pth")):
        os.remove("fine_tunned_model_and_optimizer.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    "fine_tunned_model_and_optimizer.pth")

