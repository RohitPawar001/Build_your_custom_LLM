# Custom LLM Training Project

This project focuses on training and fine-tuning custom language models using PyTorch library.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RohitPawar001/Build_your_custom_LLM.git
cd <repository-name>
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
├── fine_tunned_model_and_optimizer.pth  # Fine-tuned model weights
└── pretrained_model_and_optimizer.pth   # Pretrained model weights
```

## Usage

### Training a New Model

1. Prepare your dataset in the required format
2. Configure your training parameters
3. Run the training script:
```bash
python main.py
```

### Fine-tuning an Existing Model

1. Load a pretrained model
2. Configure fine-tuning parameters
3. Run the fine-tuning script:



## Model Files

- `pretrained_model_and_optimizer.pth`: Contains the pretrained model weights and optimizer state
- `fine_tunned_model_and_optimizer.pth`: Contains the fine-tuned model weights and optimizer state

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


