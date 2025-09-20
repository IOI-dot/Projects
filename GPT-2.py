import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
DATA_PATH = r"C:\Users\Omar\Downloads\AI-Chatbot-Conversation-Dataset-main\AI-Chatbot-Conversation-Dataset-main\dataset.csv"
MODEL_SAVE_PATH = "gpt2_chatbot"
EPOCHS = 3
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 50

df = pd.read_csv(DATA_PATH)
df["text"] = df["prompt"].astype(str) + " <|sep|> " + df["response"].astype(str)
dataset = Dataset.from_pandas(df[["text"]])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<PAD>", "sep_token": "<|sep|>"})

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # for new special tokens
model = model.to(DEVICE)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    logging_steps=100,
    prediction_loss_only=True,
    fp16=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("GPT-2 chatbot fine-tuned and saved!")
