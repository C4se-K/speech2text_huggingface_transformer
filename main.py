from prepare_data import preprocessor

import os
import soundfile as sf
from datasets import Dataset, Audio, Features, Value
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, Speech2TextConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer 
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding
import torch
from torch.optim import AdamW
from tqdm import tqdm


processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
preprocessor_instance = preprocessor("dataset_wav", processor)
data = preprocessor_instance.load_files()
dataset = preprocessor_instance.preprocess(data)


config = Speech2TextConfig()
model = Speech2TextForConditionalGeneration(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

class Cu_DataCollator(DataCollatorWithPadding):
    def __init__(self, processor):
        super().__init__(tokenizer=processor.tokenizer)
        self.processor = processor
    
    def __call__(self, features):
        input_features = [feature["input_features"].clone().detach() for feature in features]
        attention_mask = [feature["attention_mask"].clone().detach() for feature in features]
        input_ids = [feature["input_ids"].clone().detach() for feature in features]
        
        batch = {
            "input_features": torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True, padding_value=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "input_ids": self.processor.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")["input_ids"],
        }
        
        return batch
    
data_collator = Cu_DataCollator(processor=processor)


train_loader = DataLoader(
    dataset['train'],
    batch_size=32,
    shuffle=True,
    collate_fn=data_collator
)

eval_loader = DataLoader(
    dataset['test'],
    batch_size=32,
    shuffle=False,
    collate_fn=data_collator
)

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(input_features=batch["input_features"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"])

        loss = outputs.loss
        total_loss += loss.item()


        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()} 

            outputs = model(input_features=batch["input_features"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["input_ids"])

            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(data_loader)


num_epochs = 15
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    valid_loss = evaluate(model, eval_loader, device)
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {valid_loss:.4f}")

torch.save(model.state_dict(), os.path.join(os.getcwd(), "model0001.pth"))
model.save_pretrained(os.path.join(os.getcwd(), "model0001"))

