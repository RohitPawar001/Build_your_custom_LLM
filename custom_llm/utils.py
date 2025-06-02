import os
import json
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pretrain.train_dataset import GPTDataset


def get_device():
    """
    This function returns the device.
    It takes in no arguments.
    It then returns the device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device



def calc_loss_per_batch(input_batch, target_batch, model, device):
    """
    This function calculates the loss per batch.
    It takes in the input batch, target batch, model, and device.
    It then returns the loss.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    This function calculates the loss per loader.
    It takes in the data loader, model, device, and num batches.
    It then returns the loss.
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_per_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    This function evaluates the model.
    It takes in the model, train loader, val loader, device, and eval iter.
    It then returns the train loss and val loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss



def create_dataloader(text,
                      tokenizer,
                      batch_size=10,
                      max_length=256,
                      stride=128,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0):
    
    
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers
                            )
    return dataloader



def download_and_load_file(file):
    file_path = r"D:\software_2\practice\gpt"
    
    if not os.path.exists(file):
        if file.startswith("https"):
            
            with urllib.request.urlopen(file) as response:
                text_data = response.read().decode("utf-8")
            
            
            filename = os.path.basename(file) or "downloaded_file.json"
            full_path = os.path.join(file_path, filename)
            
            
            os.makedirs(file_path, exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as data_file:
                data_file.write(text_data)
            
            
            with open(full_path, "r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            return data
        else:
            raise FileNotFoundError("File does not exist. Please provide a valid URL or correct file path.")
    else:
        if file.endswith(".json"):
            with open(file, "r", encoding="utf-8") as data_file:
                data = json.load(data_file)
        else:
            with open(file, "r", encoding="utf-8") as data_file:
                data = data_file.read()
        return data
    

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def save_pretrained_model(model, optimizer, model_name):
    if os.path.exists(os.path.join(os.getcwd(),f"{model_name}_pretrained_model.pth")):
        os.remove(f"{model_name}_pretrained_model.pth")

    torch.save({
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict()
    },
    f"{model_name}_pretrained_model.pth")
    print("pretrained model has been saved successfully")

def partition_data(data):
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion]

    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    return train_data, test_data, val_data


def load_fine_tunned_model(model, device, model_name):
    if os.path.exists(os.path.join(os.getcwd(),f"instruct_{model_name}.pth")):
        checkpoint = torch.load(f"instruct_{model_name}.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model
    else:
        raise FileNotFoundError("There is no finetunned model you need to finetunne the model")

def load_pretrained_model(model, device, model_name):
    if os.path.exists(os.path.join(os.getcwd(),f"{model_name}_pretrained_model.pth")):
        checkpoint = torch.load(f"{model_name}_pretrained_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model
    else:
        raise FileNotFoundError("There is no pretraind model you need to train the model")