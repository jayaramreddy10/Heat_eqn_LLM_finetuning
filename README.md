# Solving Heat Equation Using Finite Difference Method

This project demonstrates the solution of the heat equation using the finite difference method. It also includes data preparation for LLM fine-tuning with four boundary cases, fine-tuning the Granite model, and inferring results using the fine-tuned model.

## Repository Structure

- `data_preparation.py`: Script for preparing data for fine-tuning based on four boundary cases.
- `finetune_granite.py`: Script for fine-tuning the Granite LLM model using the prepared data.
- `infer_granite.py`: Script for inferring results using the fine-tuned Granite model.
- `requirements.txt`: Dependencies required to execute the scripts.
- `README.md`: This file.

## Getting Started

### Prerequisites

1. Install Python (>= 3.8).
2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Modify the boundary conditions and heat equation parameters in `data_preparation.py` as needed.
2. Run the script to generate the dataset for fine-tuning:

   ```bash
   python data_preparation.py
   ```
3. The output will be saved to a text file (e.g., `fine_tuning_data.txt`).

### Fine-Tuning the Granite Model

1. Ensure that `fine_tuning_data.txt` is available in the project directory.
2. Specify the path to the dataset and model in `finetune_granite.py`.
3. Run the script to fine-tune the model:

   ```bash
   python finetune_granite.py
   ```
4. The fine-tuned model and tokenizer will be saved to the `results/` folder.

### Inferring with the Fine-Tuned Model

1. Modify `infer_granite.py` to specify the input query for the fine-tuned model.
2. Run the script:

   ```bash
   python infer_granite.py
   ```
3. The model's response will be displayed in the terminal.

## File Descriptions

### `data_preparation.py`
Generates a dataset for fine-tuning the LLM based on the finite difference method for solving the heat equation. It handles four boundary cases:
- Fixed temperature at both ends.
- Insulated boundaries.
- Linearly varying boundary conditions.
- Mixed boundary conditions.

### `finetune_granite.py`
Fine-tunes the Granite LLM using LoRA with the dataset generated in `data_preparation.py`. The model is configured for 4-bit quantization and optimized for causal language modeling tasks.

### `infer_granite.py`
Loads the fine-tuned Granite model and provides inference capabilities based on user queries.

### `requirements.txt`
Specifies the dependencies needed for the project, including `transformers`, `peft`, `torch`, `datasets`, and `nltk`.

## Additional Notes

- Ensure you have a GPU-enabled environment for efficient training and inference.
- The fine-tuning script uses LoRA for parameter-efficient fine-tuning, reducing the need for large computational resources.
- Model and tokenizer are saved in the `results/` folder. You can reload them using the following code snippet:

  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model = AutoModelForCausalLM.from_pretrained("results")
  tokenizer = AutoTokenizer.from_pretrained("results")
  ```

