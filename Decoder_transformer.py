import re
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader



DATA_PATH = r"C:\Users\Omar\Downloads\AI-Chatbot-Conversation-Dataset-main\dataset.csv"
MAX_LEN = 30
BATCH_SIZE = 32
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
LR = 1e-3
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

class Vocab:
    def __init__(self, texts, max_size=5000):
        from collections import Counter
        counter = Counter([t for txt in texts for t in tokenize(txt)])
        words = [w for w,_ in counter.most_common(max_size)]
        self.itos = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + words
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, tokens):
        ids = [self.stoi["<BOS>"]] + [self.stoi.get(t,1) for t in tokens] + [self.stoi["<EOS>"]]
        return ids[:MAX_LEN] + [self.stoi["<PAD>"]] * (MAX_LEN - len(ids))

class ChatDataset(Dataset):
    def __init__(self, prompts, responses, vocab):
        self.samples = []
        for q,a in zip(prompts,responses):
            tokens = tokenize(f"user: {q} assistant: {a}")
            self.samples.append(vocab.encode(tokens))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]
        return torch.tensor(x)

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=128
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, vocab_size)
    def forward(self, x):
        emb = self.embed(x)
        out = self.decoder(emb, emb)
        return self.fc(out)

df = pd.read_csv(DATA_PATH)
prompts = df["prompt"].astype(str).tolist()
responses = df["response"].astype(str).tolist()

vocab = Vocab(prompts + responses)
dataset = ChatDataset(prompts, responses, vocab)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = DecoderOnlyTransformer(len(vocab.itos)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

for epoch in range(1,EPOCHS+1):
    model.train()
    for xb in train_loader:
        xb = xb.T.to(DEVICE)
        inp = xb[:-1,:]
        target = xb[1:,:]

        opt.zero_grad()
        out = model(inp)
        loss = loss_fn(out.reshape(-1, out.shape[-1]), target.reshape(-1))
        loss.backward()
        opt.step()
    print(f"Epoch {epoch} | Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "decoder_only_chatbot.pth")
torch.save(vocab, "decoder_only_vocab.pth")
