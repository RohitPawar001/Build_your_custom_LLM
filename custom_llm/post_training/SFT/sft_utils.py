import os
import urllib
import json
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
    This file contains the implementation of the format_input and custom_collate_fn functions.
    this is the main utils for the custom LLM.
"""

def format_input(entry):
    """
    This function formats the input for the custom LLM.
    It takes in the entry and returns the formatted input.
    """
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
    """
    This function collates the batch for the custom LLM.
    It takes in the batch, pad token id, ignore index, allowed max length, and device.
    It then pads and stacks the inputs and targets.
    """
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






def save_fine_tunned_model(model, optimizer,model_name):
    
    """
    This function saves the fine-tuned model and optimizer.
    It takes in the model and optimizer.
    It then saves the model and optimizer.
    """

    if os.path.exists(os.path.join(os.getcwd(),f"instruct_{model_name}.pth")):
        os.remove(f"instruct_{model_name}.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    f"instruct_{model_name}.pth")

