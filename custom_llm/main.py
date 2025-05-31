import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_args import model_arguments
from model.model import LLMModel
from pretrain.train import train_model
from utils import create_dataloader
from post_training.SFT.sft_dataset import SFTDataloaders
from post_training.SFT.sft_trainer import FineTuneModel
from post_training.SFT.sft_utils import save_fine_tunned_model
from generate import (generate_text_simple,
                      text_to_token_ids,
                      token_ids_to_text)
from utils import (
    get_device,
    save_pretrained_model,
    load_pretrained_model,
    load_fine_tunned_model,
    download_and_load_file,
    partition_data
)


def interactive_generation(model, tokenizer, device):
    """Handle interactive text generation with the model"""
    model.eval()
    while True:
        choice = input("Do you want to continue? (y/n): ").lower()
        if choice == "y":
            text = input("Enter your prompt: ")
            
            torch.manual_seed(123)
            token_ids = generate_text_simple(
                model=model,
                idx=text_to_token_ids(text, tokenizer),
                max_new_tokens=23,
                context_size=model_arguments["context_length"]
            )
            print("Generated text:", token_ids_to_text(token_ids, tokenizer))
        else:
            break


def train_custom_model():
    """Train a custom model from scratch"""
    print("Initializing the model...")
    
    # Initialize models and components
    model = LLMModel(model_arguments)
    base_model = LLMModel(model_arguments)
    device = get_device()
    print(f"Using device: {device}")
    
    # Move models to device
    model = model.to(device)
    base_model = base_model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 1

    # Pre-training phase
    data_file = input("Enter the file path/URL for pre-training data: ")
    data = download_and_load_file(data_file)
    print("Data file downloaded")
    
    train_data, test_data, val_data = partition_data(data)
    
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=2,
        max_length=model_arguments["context_length"],
        stride=model_arguments["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=2,
        max_length=model_arguments["context_length"],
        stride=model_arguments["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print("Initializing training...")
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device, 
        num_epochs, eval_freqs=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    print("Pre-training complete")

    save_pretrained_model(model, optimizer)
    print("Pre-trained model saved")
    
    # Fine-tuning phase
    print("Loading pre-trained model for fine-tuning...")
    model = load_pretrained_model(base_model, device)
    model = model.to(device)
    
    print("Preparing model for fine-tuning...")
    sft_data_file = input("Enter the file path/URL for fine-tuning data: ")
    sft_data = download_and_load_file(sft_data_file)
    print("Fine-tuning data downloaded")
    
    sft_train_data, sft_test_data, sft_val_data = partition_data(sft_data)
    
    train_dataloader, val_dataloader, test_dataloader = SFTDataloaders(
        train_data=sft_train_data, 
        val_data=sft_val_data, 
        test_data=sft_test_data, 
        tokenizer=tokenizer,
        device=device
    ).data_loaders()
    
    sft = FineTuneModel(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        tokenizer=tokenizer,
        val_data=sft_val_data,
    )
    
    print("Starting supervised fine-tuning...")
    model = sft.train()
    save_fine_tunned_model(model, optimizer)
    print("Fine-tuned model has been saved")
    
    return model


def load_existing_model():
    """Load an existing fine-tuned model"""
    base_model = LLMModel(model_arguments)
    device = get_device()
    print(f"Using device: {device}")
    
    base_model = base_model.to(device)
    model = load_fine_tunned_model(base_model, device)
    model = model.to(device)
    print("Fine-tuned model loaded")
    
    return model


def main():
    """Main function to handle user choice and execute appropriate workflow"""
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    
    print("Welcome to the LLM Training and Inference System!")
    print("1. Use existing fine-tuned model")
    print("2. Train a custom model from scratch")
    
    try:
        usecase = int(input("Please select an option (1 or 2): "))
        
        if usecase == 1:
            model = load_existing_model()
            interactive_generation(model, tokenizer, device)
            
        elif usecase == 2:
            model = train_custom_model()
            print("Loading fine-tuned model for inference...")
            base_model = LLMModel(model_arguments).to(device)
            model = load_fine_tunned_model(base_model, device)
            interactive_generation(model, tokenizer, device)
            
        else:
            raise ValueError("Invalid option selected. Please choose 1 or 2.")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run the script again and select a valid option.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()