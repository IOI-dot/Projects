import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

DATA_PATH = r"C:\Users\Omar\Downloads\archive (17)\emotion_sentimen_dataset.csv"
MODEL_NAME = "bert-base-uncased"
MODEL_SAVE_PATH = "bert_emotion"
EPOCHS = 3
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(DATA_PATH)
texts = df["text"].astype(str).tolist()
labels_raw = df["Emotion"].astype(str).tolist()

le = LabelEncoder()
labels = le.fit_transform(labels_raw)

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
df_train = pd.DataFrame({"text": X_train, "label": y_train})
df_val = pd.DataFrame({"text": X_val, "label": y_val})
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_)).to(DEVICE)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=True if DEVICE == "cuda" else False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("BERT emotion classifier fine-tuned and saved")
