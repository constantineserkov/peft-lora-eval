# test_data_loader.py
import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.dataloader import (
    format_prompt,
    tokenize,
    add_length,
    get_cleaned_sorted_dataset,
    get_dataloader,
    DataCollatorForCustomPadding,
)


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


def test_tokenize_and_labels(tokenizer, small_dataset):
    batch = small_dataset.add_column("prompt", ["Prompt1", "Prompt2"])[:2]
    result = tokenize(batch, tokenizer)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    # labels should have -100 for prompt tokens
    assert all(isinstance(x, list) for x in result["labels"])
    assert all(-100 in x for x in result["labels"])


def test_add_length(tokenizer, small_dataset):
    # tokenize first so input_ids exist
    batch = small_dataset.add_column("prompt", ["Prompt1", "Prompt2"])[:2]
    tokenized = tokenize(batch, tokenizer)
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
    tokenized = tmp_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
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
