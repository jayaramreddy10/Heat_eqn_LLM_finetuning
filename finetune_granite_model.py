import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
# from trl import SFTTrainer
from nltk.translate.meteor_score import meteor_score
# from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq

class finetune_auto:
    def __init__(self, granite_model, input_text_file):
        # Model and dataset setup
        self.model_name = granite_model
        self.input_text_file = input_text_file
        
        # Training settings
        self.max_seq_length = 512
        self.packing = False
        self.device_map = {"": 0}
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = 'float16'
        self.bnb_4bit_quant_type = 'nf4'
        self.use_nested_quant = False
        
        self.output_dir = "./finetuned_granite_model"
        self.num_train_epochs = 3
        self.fp16 = False
        self.bf16 = False
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.max_grad_norm = 0.3
        self.learning_rate = 1e-3
        self.weight_decay = 0.001
        self.optim = "adamw_hf"
        self.lr_scheduler_type = "cosine_with_restarts"
        self.max_steps = -1
        self.warmup_ratio = 0.03
        self.group_by_length = False
        self.save_steps = 100
        self.logging_steps = 25
        
        # LoRA settings
        self.lora_r = 64
        self.lora_alpha = 32
        self.lora_dropout = 0.075
        
        # Internal attributes
        self.dataset_object = None
        self.transformed_dataset = None
        self.train_dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
        )
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def prepare_dataset_from_text_file(self):
        if not os.path.exists(self.input_text_file):
            raise FileNotFoundError(f"Text file '{self.input_text_file}' not found.")

        with open(self.input_text_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Prepare data for training
        data = [{"text": line.strip()} for line in lines if line.strip()]
        dataset = Dataset.from_list(data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
            )

        self.transformed_dataset = dataset.map(tokenize_function, batched=True)
        self.train_dataset = self.transformed_dataset

    def setup_trainer(self):
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            report_to="tensorboard",
        )
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            peft_config=peft_config,
            # dataset_text_field="text",
            # max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            # packing=self.packing,
        )

        # self.trainer = SFTTrainer(
        #     model = self.model,
        #     tokenizer = self.tokenizer,
        #     train_dataset=self.train_dataset,
        #     data_collator = DataCollatorForSeq2Seq(tokenizer = self.tokenizer),
        #     args = SFTConfig(
        #         per_device_train_batch_size=2,
        #         gradient_accumulation_steps=4,
        #         warmup_steps = 5,
        #         num_train_epochs = 3, # Set this for 1 full training run.
        #         #max_steps = 60,
        #         learning_rate = 2e-4,
        #         fp16=self.fp16,
        #         bf16=self.bf16,
        #         optim = "adamw_8bit",
        #         weight_decay = 0.01,
        #         lr_scheduler_type = "linear",
        #         seed = 3407,
        #         output_dir = "model_traning_outputs",
        #         report_to = "none",
        #         max_seq_length = 2048,
        #         dataset_num_proc = 4,
        #         packing = False, # Can make training 5x faster for short sequences.
        #     ),
        # )


    def train(self):
        self.trainer.train()

# Example usage
granite_model = "ibm-granite/granite-3.1-2b-base"
input_text_file = '/home/jayaram/research_threads/basic_architectures/heat_eqn/fine_tuning_data.txt'

finetuner = finetune_auto(granite_model, input_text_file)

finetuner.load_model_and_tokenizer()
finetuner.prepare_dataset_from_text_file()
finetuner.setup_trainer()
finetuner.train()
