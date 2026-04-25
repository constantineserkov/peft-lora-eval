# test_data_loader.py
import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.dataloader import (
    format_prompt,
    load_alpaca_data,
    tokenize_supervised_causal_lm_batch,
    add_length,
    get_cleaned_sorted_dataset,
    get_dataloader,
    DataCollatorForCustomPadding,
)


class _LoadedDataset:
    def __init__(self):
        self.column_names = ["instruction", "input", "output"]

    def remove_columns(self, column):
        assert column == "input"
        return self

    def __len__(self):
        return 2


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def small_dataset():
    # Minimal dataset with instruction + output
    return Dataset.from_dict({
        "instruction": ["Say hello", "Count to three"],
        "output": ["Hello!", "1, 2, 3"],
    })


def test_format_prompt(small_dataset):
    batch = small_dataset[:2]
    result = format_prompt(batch)
    assert "prompt" in result
    assert len(result["prompt"]) == 2
    assert "###Instruction:" in result["prompt"][0]


def test_load_alpaca_data_uses_full_train_split_when_subset_is_none(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _LoadedDataset()

    monkeypatch.setattr("src.dataloader.load_dataset", fake_load_dataset)

    load_alpaca_data("yahma/alpaca-cleaned", None)

    assert calls["args"] == ("yahma/alpaca-cleaned",)
    assert calls["kwargs"]["split"] == "train"


def test_load_alpaca_data_uses_train_slice_when_subset_is_set(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _LoadedDataset()

    monkeypatch.setattr("src.dataloader.load_dataset", fake_load_dataset)

    load_alpaca_data("yahma/alpaca-cleaned", 100)

    assert calls["args"] == ("yahma/alpaca-cleaned",)
    assert calls["kwargs"]["split"] == "train[:100]"


def test_tokenize_and_labels(tokenizer, small_dataset):
    batch = small_dataset.add_column("prompt", ["Prompt1", "Prompt2"])[:2]
    result = tokenize_supervised_causal_lm_batch(batch, tokenizer)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    # labels should have -100 for prompt tokens
    assert all(isinstance(x, list) for x in result["labels"])
    assert all(-100 in x for x in result["labels"])


def test_add_length(tokenizer, small_dataset):
    # tokenize first so input_ids exist
    batch = small_dataset.add_column("prompt", ["Prompt1", "Prompt2"])[:2]
    tokenized = tokenize_supervised_causal_lm_batch(batch, tokenizer)
    example = {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
        "labels": tokenized["labels"][0],
    }
    result = add_length(example)
    assert "length" in result
    assert isinstance(result["length"], int)


def test_get_cleaned_sorted_dataset(tokenizer, small_dataset):
    # prepare dataset with length column
    prompts = format_prompt(small_dataset[:2])
    tmp_ds = small_dataset.add_column("prompt", prompts["prompt"])
    tokenized = tmp_ds.map(lambda b: tokenize_supervised_causal_lm_batch(b, tokenizer), batched=True)
    ds = tokenized.map(add_length)
    cleaned = get_cleaned_sorted_dataset(ds)
    # check columns dropped
    for col in ["instruction", "output", "length", "prompt"]:
        assert col not in cleaned.column_names


# def test_get_dataloader(tokenizer, small_dataset):
#     prompts = format_prompt(small_dataset[:2])
#     tmp_ds = small_dataset.add_column("prompt", prompts["prompt"])
#     tokenized = tmp_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
#     ds = tokenized.map(add_length)
#     cleaned = get_cleaned_sorted_dataset(ds)
#
#     dataloader = get_dataloader(cleaned, tokenizer, batch_size=2, seed=42)
#     batch = next(iter(dataloader))
#     assert "input_ids" in batch
#     assert "attention_mask" in batch
#     assert "labels" in batch
#     assert isinstance(batch["input_ids"], torch.Tensor)
