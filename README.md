# Solving Heat Equation Using Finite Difference Method

This project demonstrates the solution of the heat equation using the finite difference method. It also includes data preparation for LLM fine-tuning with four boundary cases, fine-tuning the Granite model, and inferring results using the fine-tuned model.

## Repository Structure
- `heat_eqn_handling_boundary_conditions.py`: Script for solving heat eqn in 2D using Finite difference method for all the 4 boundary conditions.
- `generate_dataset.py`: Script for preparing data for fine-tuning based on four boundary cases.
- `finetune_granite_model.py`: Script for fine-tuning the Granite LLM model using the prepared data.
- `inference_Granite.py`: Script for inferring results using the fine-tuned Granite model.
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

1. Modify the boundary conditions and heat equation parameters in `generate_dataset.py` as needed.
2. Run the script to generate the dataset for fine-tuning:

   ```bash
   python generate_dataset.py
   ```
3. The output will be saved to a text file (e.g., `fine_tuning_data.txt`).

### Fine-Tuning the Granite Model

1. Ensure that `fine_tuning_data.txt` is available in the project directory.
2. Specify the path to the dataset and model in `finetune_granite_model.py`.
3. Run the script to fine-tune the model:

   ```bash
   python finetune_granite_model.py
   ```
4. The fine-tuned model and tokenizer will be saved to the `finetuned_granite_model/` folder.

### Inferring with the Fine-Tuned Model

1. Modify `inference_Granite.py` to specify the input query for the fine-tuned model.
2. Run the script:

   ```bash
   python inference_Granite.py
   ```
3. The model's response will be displayed in the terminal.

## File Descriptions

### `generate_dataset.py`
Generates a dataset for fine-tuning the LLM based on the finite difference method for solving the heat equation. It handles four boundary cases:
- Fixed temperature at both ends.
- Insulated boundaries.
- Linearly varying boundary conditions.
- Mixed boundary conditions.

### `finetune_granite_model.py`
Fine-tunes the Granite LLM using LoRA with the dataset generated in `generate_dataset.py`. The model is configured for 4-bit quantization and optimized for causal language modeling tasks.

### `inference_Granite.py`
Loads the fine-tuned Granite model and provides inference capabilities based on user queries.

### `requirements.txt`
Specifies the dependencies needed for the project, including `transformers`, `peft`, `torch`, `datasets`, and `nltk`.

## Additional Notes

- Ensure you have a GPU-enabled environment for efficient training and inference.
- The fine-tuning script uses LoRA for parameter-efficient fine-tuning, reducing the need for large computational resources.
- Model and tokenizer are saved in the `finetuned_granite_model/` folder. You can reload them using the following code snippet:


