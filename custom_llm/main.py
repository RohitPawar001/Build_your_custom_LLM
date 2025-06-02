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
    

    # Pre-training phase
    model_name = input("Enter the name of the model you want to build: ")
    data_file = input("Enter the file path/URL for pre-training data (for example: https://www.gutenberg.org/files/11/11-0.txt/ or /data/train.txt): ")
    data = download_and_load_file(data_file)
    print("Data file downloaded/ loaded")
    batch_size = int(input("Enter the batch size for pre-training: "))
    num_epochs = int(input("Enter the number of epochs for pre-training: "))
    
    train_data, test_data, val_data = partition_data(data)
    
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=batch_size,
        max_length=model_arguments["context_length"],
        stride=model_arguments["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=batch_size,
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
        start_context="The object of our misery", tokenizer=tokenizer
    )
    print("Pre-training complete")

    save_pretrained_model(model, optimizer, model_name)
    print("Pre-trained model saved")
    
    # Fine-tuning phase
    print()
    print("-------------------------------------------------")
    print("Supervised fine-tuning the model...")
    print("-------------------------------------------------")
    print()
    print("Loading pre-trained model for fine-tuning...")

    model = load_pretrained_model(base_model, device, model_name)
    model = model.to(device)
    
    
    print("Preparing model for fine-tuning...")
    sft_data_file = input("Enter the file path/URL for fine-tuning data (for example: https://www.gutenberg.org/files/11/11-0.json/ or /data/instruction_dataset.json): ")
    sft_data = download_and_load_file(sft_data_file)
    print("Fine-tuning data downloaded/ loaded")
    num_epochs = int(input("Enter the number of epochs for fine-tuning: "))
    
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
    save_fine_tunned_model(model, optimizer, model_name)
    print("Fine-tuned model has been saved")

    print()
    print("-------------------------------------------------")
    print("Supervised fine-tuning complete model is ready for inference")
    print("-------------------------------------------------")
    
    return model, model_name


def load_existing_model(model_loader, model_name):
    """Load an existing fine-tuned model"""
    base_model = LLMModel(model_arguments)
    device = get_device()
    print(f"Using device: {device}")
    
    base_model = base_model.to(device)
    model = model_loader(base_model, device, model_name)
    model = model.to(device)
    print("Model loaded")
    
    return model


def finetune_pretrained_model():
    """Fine-tune an existing pretrained model"""
    device = get_device()
    print(f"Using device: {device}")
    
    # Get model name and load pretrained model
    model_name = input("Enter the name of the pretrained model to fine-tune: ")
    base_model = LLMModel(model_arguments)
    base_model = base_model.to(device)
    
    print("Loading pretrained model...")
    model = load_pretrained_model(base_model, device, model_name)
    model = model.to(device)
    print("Pretrained model loaded successfully")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    
    # Fine-tuning setup
    print()
    print("-------------------------------------------------")
    print("Setting up fine-tuning...")
    print("-------------------------------------------------")
    
    sft_data_file = input("Enter the file path/URL for fine-tuning data (for example: https://www.gutenberg.org/files/11/11-0.json/ or /data/instruction_dataset.json): ")
    sft_data = download_and_load_file(sft_data_file)
    print("Fine-tuning data downloaded/loaded")
    batch_size = int(input("Enter the batch size for fine-tuning: "))
    num_epochs = int(input("Enter the number of epochs for fine-tuning: "))
    
    
    
    sft_train_data, sft_test_data, sft_val_data = partition_data(sft_data)
    
    train_dataloader, val_dataloader, test_dataloader = SFTDataloaders(
        train_data=sft_train_data, 
        val_data=sft_val_data, 
        test_data=sft_test_data, 
        tokenizer=tokenizer,
        batch_size=batch_size,
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
    
    # Save the fine-tuned model
    finetuned_model_name = input("Enter a name for the fine-tuned model (or press Enter to use original name): ")
    if not finetuned_model_name.strip():
        finetuned_model_name = model_name
    
    save_fine_tunned_model(model, optimizer, finetuned_model_name)
    print("Fine-tuned model has been saved")

    print()
    print("-------------------------------------------------")
    print("Fine-tuning complete! Model is ready for inference")
    print("-------------------------------------------------")
    
    return model, finetuned_model_name


def main():
    """Main function to handle user choice and execute appropriate workflow"""
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    
    print("-------------------------------------------------")
    print("Welcome to the LLM Training and Inference System!")
    print("-------------------------------------------------")
    print()
    print("1. Use pretrained model")
    print("2. Use existing fine-tuned model")
    print("3. Train a custom model from scratch")
    print("4. Fine-tune an existing pretrained model")
    
    try:
        usecase = int(input("Please select an option (1, 2, 3, or 4): "))
        
        if usecase == 1:
            print()
            print("-------------------------------------------------")
            print("Loading pretrained model...")
            print("-------------------------------------------------")
            model_name = input("Enter the name of the pretrained model to load: ")
            model = load_existing_model(load_pretrained_model, model_name)
            interactive_generation(model, tokenizer, device)

        elif usecase == 2:
            print()
            print("-------------------------------------------------")
            print("Loading fine-tuned model...")
            print("-------------------------------------------------")
            model_name = input("Enter the name of the fine-tuned model to load: ")
            model = load_existing_model(load_fine_tunned_model, model_name)
            interactive_generation(model, tokenizer, device)
            
        elif usecase == 3:
            print()
            print("-------------------------------------------------")
            print("Training custom model...")
            print("-------------------------------------------------")
            model, model_name = train_custom_model()
            base_model = LLMModel(model_arguments).to(device)
            model = load_fine_tunned_model(base_model, device, model_name)
            interactive_generation(model, tokenizer, device)
            
        elif usecase == 4:
            print()
            print("-------------------------------------------------")
            print("Fine-tuning pretrained model...")
            print("-------------------------------------------------")
            model, model_name = finetune_pretrained_model()
            interactive_generation(model, tokenizer, device)
            
        else:
            raise ValueError("Invalid option selected. Please choose 1, 2, 3, or 4.")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run the script again and select a valid option.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()