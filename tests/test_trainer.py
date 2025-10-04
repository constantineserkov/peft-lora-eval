# Testing: Create tests/test_trainer.py to verify train_model
# (e.g., check loss computation, gradient updates).

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import wandb
import pytest
from src.trainer import train_model


# Dummy Model for testing
class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10, seq_len=5):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 64)
        self.linear = torch.nn.Linear(64, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.embed(input_ids)
        logits = self.linear(embeds)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return type('Output', (object,), {'loss': loss, 'logits': logits})()

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples=10, seq_len=5, vocab_size=10):
        self.data = []
        for _ in range(num_samples):
            input_ids = torch.randint(0, vocab_size, (seq_len,))
            self.data.append({
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
                'labels': input_ids.clone()
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

@pytest.fixture
def setup_training_components():
    # Minimal config
    config_dict = {
        'training': {
            'lr': 1e-4,
            'optimizer': {
                'weight_decay': 0.01,
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'scheduler': {
                'num_warmup_steps': 1
            },
            'num_training_steps': 5,
            'num_epochs': 1,
            'grad_accumulation_steps': 1,
        },
        'output_path': './test_output',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DummyModel().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False)  # Dummy tokenizer, use_fast=False to avoid warnings
    train_dataset = DummyDataset(num_samples=4)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataset = DummyDataset(num_samples=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    return model, tokenizer, train_loader, device, config_dict, val_loader

def test_train_model(setup_training_components, monkeypatch):
    # Disable wandb for test
    wandb.init(mode="disabled")

    model, tokenizer, train_loader, device, config_dict, val_loader = setup_training_components

    # Run the trainer and check if it completes without errors
    try:
        train_model(model, tokenizer, train_loader, device, config_dict, val_loader=val_loader)
        assert True  # If no exception, test passes
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")