import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split

import wandb

# Start a W&B
wandb.login()
run = wandb.init(project="T5_Paraphrase_Paws")

# Model and data configuration
config = {
    "model_name": "Vamsi/T5_Paraphrase_Paws",
    "data_file": "./data/quora_duplicate_questions.tsv",
    "max_length": 32,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_train_epochs": 2,
    "output_dir": "./finetuned_paraphraser",
}
wandb.config.update(config)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Create dataset class for Quora dataset
class QuoraParaphraseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, prefix="paraphrase: ", max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        question1 = row['question1']
        question2 = row['question2']
        
        # Add prefix for T5
        input_text = self.prefix + question1
        target_text = question2
        
        # Tokenize inputs and targets
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension introduced by tokenizer
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def main():
    # Load and prepare dataset
    df = pd.read_csv(config["data_file"], sep='\t')
    
    # Filter only duplicate questions
    duplicate_df = df[df['is_duplicate'] == 1].reset_index(drop=True)

    # Take 10% of the data
    duplicate_df = duplicate_df.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Keep the rows with tokens <= 32
    duplicate_df['question1_token_len'] = duplicate_df['question1'].apply(lambda x: len(x.split()))
    duplicate_df['question2_token_len'] = duplicate_df['question2'].apply(lambda x: len(x.split()))
    duplicate_df = duplicate_df[
        (duplicate_df['question1_token_len'] <= 32) &
        (duplicate_df['question2_token_len'] <= 32)
    ]
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(duplicate_df, test_size=0.2, random_state=42)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # Save the train and validation datasets to CSV files
    train_df.to_csv("./data/train.csv", index=False)
    val_df.to_csv("./data/val.csv", index=False)
    
    # Load tokenizer and model
    print(f"Loading model and tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])
    
    # Prepare datasets
    train_dataset = QuoraParaphraseDataset(train_df, tokenizer, max_length=config['max_length'])
    val_dataset = QuoraParaphraseDataset(val_df, tokenizer, max_length=config['max_length'])
    
    # Set up training arguments
    # Check here: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate= config['learning_rate'],
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=100,
        report_to="wandb"
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save final model and tokenizer
    print(f"Saving model to {config['output_dir']}")
    model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    print("Fine-tuning completed successfully!")
    run.finish()

if __name__ == "__main__":
    main()
