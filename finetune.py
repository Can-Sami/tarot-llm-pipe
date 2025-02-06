from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
from transformers import TrainingArguments
import os

def prepare_dataset():
    # Load your dataset - modify this according to your data format
    dataset = load_dataset("json", data_files="dataset_entries.json")
    
    # Format the dataset for instruction tuning
    def format_instruction(example):
        return {
            "text": f"### Instruction: {example['instruction']}\n\n### Response: {example['response']}"
        }
    
    dataset = dataset.map(format_instruction)
    return dataset

def main():
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ollama/llama2:3.2",
        max_seq_length=2048,
        dtype=None,  # defaults to float16
        load_in_4bit=True,  # Quantization for memory efficiency
    )

    # Prepare the model for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # rank for LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )

    # Prepare dataset
    dataset = prepare_dataset()

    # Initialize trainer
    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=training_args,
        max_seq_length=2048,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("./finetuned_model/final")

if __name__ == "__main__":
    main()
