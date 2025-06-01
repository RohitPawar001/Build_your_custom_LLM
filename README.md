# Custom LLM Training and Fine-tuning System

This project provides a system for training, fine-tuning, and using custom language models. It supports both pre-training from scratch and fine-tuning of existing models.

## Features

- Pre-train models from scratch
- Fine-tune existing pre-trained models
- Use pre-trained or fine-tuned models for inference
- Interactive text generation
- Support for custom datasets

## Dataset Formats

### Fine-tuning Dataset Format

The fine-tuning dataset should be in JSON format with the following structure:

```json
{
    "instruction": "The task or instruction to be performed",
    "input": "The input text or prompt",
    "output": "The expected output or response"
}
```

Example:
```json
{
    "instruction": "Evaluate the following phrase by transforming it into the spelling given.",
    "input": "freind --> friend",
    "output": "The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\"."
}
```

### Pre-training Dataset Format

For pre-training, you can use:
- Plain text files (.txt)
- Text data from URLs (e.g., Gutenberg books)

## Usage

1. Run the main script:
```bash
python main.py
```

2. Choose from the following options:
   - Use pretrained model
   - Use existing fine-tuned model
   - Train a custom model from scratch
   - Fine-tune an existing pretrained model

3. Follow the prompts to:
   - Enter model names
   - Provide dataset paths/URLs
   - Set training parameters (batch size, epochs)
   - Interact with the model

## Model Files

The system uses the following file naming conventions:
- Pre-trained models: `{model_name}_pretrained_model.pth`
- Fine-tuned models: `instruct_{model_name}.pth`

## Requirements

- Python 3.x
- PyTorch
- tiktoken
- CUDA (optional, for GPU acceleration)

## Notes

- Make sure your dataset follows the specified format
- Model files should be in the current working directory
- For fine-tuning, ensure you have a pre-trained model available
- The system automatically handles device selection (CPU/GPU)

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RohitPawar001/Build_your_custom_LLM.git
cd custom_llm
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv AI_env
.\AI_env\Scripts\activate

# Linux/Mac
python -m venv AI_env
source AI_env/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── requirements.txt
├── .gitignore
├── instruct_{model_name}.pth  # Fine-tuned model weights
└── {model_name}_pretrained_model.pth   # Pretrained model weights
```

## Dependencies

The project requires the following main packages:
- PyTorch (>=2.0.0)
- NumPy (>=1.21.0)
- Pandas (>=1.3.0)
- scikit-learn (>=1.0.0)
- Matplotlib (>=3.4.0)
- tqdm (>=4.62.0)
- datasets (>=2.12.0)
- accelerate (>=0.20.0)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


