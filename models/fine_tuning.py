import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from tqdm import tqdm

CONFIG = {
    "model_path": "models/bert-base-uncased",
    "batch_size": 32,
    "lr": 1e-4,
    "temperature": 0.3,
    "max_length": 512,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class CoTDataset(Dataset):
    def __init__(self, file_path, tokenizer):

        df = pd.read_csv(file_path)

        if not all(col in df.columns for col in ['correct_reasoning', 'incorrect_reasoning']):
            raise ValueError("CSV must contain 'correct_reasoning' and 'incorrect_reasoning' columns")
        
        self.data = df[['correct_reasoning', 'incorrect_reasoning']].to_dict('records')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            "anchor": item["correct_reasoning"],
            "positive1": item["incorrect_reasoning"],
            "positive2": item["correct_reasoning"]
        }

class ContrastiveBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(CONFIG["model_path"])
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG["model_path"])

    def forward(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        ).to(CONFIG["device"])
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0]

def collate_fn(batch):

    return {
        "anchor": [item["anchor"] for item in batch],
        "positive1": [item["positive1"] for item in batch],
        "positive2": [item["positive2"] for item in batch]
    }

def contrastive_loss(anchor, positives, temperature):
    batch_size = anchor.size(0)
    logits = torch.matmul(anchor, positives.T) / temperature
    labels = torch.arange(0, batch_size * 2, 2, device=CONFIG["device"])
    return torch.nn.functional.cross_entropy(logits, labels)

def train():
    model = ContrastiveBERT().to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])


    dataset = CoTDataset("models/fine_llama3.csv", model.tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            anchor_embs = model(batch["anchor"])
            pos1_embs = model(batch["positive1"])
            pos2_embs = model(batch["positive2"])

            positives = torch.cat([pos1_embs, pos2_embs], dim=0)
            loss = contrastive_loss(anchor_embs, positives, CONFIG["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "models/finetuned_for_llama3.bin")

if __name__ == "__main__":
    train()
