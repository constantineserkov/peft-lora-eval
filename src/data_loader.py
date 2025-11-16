import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List, Dict, Optional
from src.logger import get_logger

logger = get_logger()


def load_alpaca_data(dataset_name: str, data_subset: int) -> Dataset:
    logger.debug("Loading the alpaca dataset")
    # Load dataset
    dataset = load_dataset(dataset_name, split=f"train[:{data_subset}]", streaming=False, cache_dir="../data/processed").remove_columns('input')

    # Log the dataset size
    logger.info(f"Loaded dataset '{dataset_name}' with {len(dataset)} examples.")

    return dataset


def format_prompt(batch) -> Dict:
    prompts = []
    for inst in batch['instruction']:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes " \
                 f"the request.\n\n###Instruction: \n{inst}\n\n###Response: \n"
        prompts.append(prompt)
    return {'prompt': prompts}


def tokenize(batch, tokenizer: PreTrainedTokenizerBase) -> Dict:
    if tokenizer.eos_token is None:
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} does not define an eos_token. "
            "Use a causal LM tokenizer (e.g., LLaMA, GPT2)."
        )
    full_texts = [prompt + output + tokenizer.eos_token for prompt, output in zip(batch['prompt'], batch['output'])]
    tokenized_all = tokenizer(full_texts)
    tokenized_prompts = tokenizer(batch['prompt'])
    prompt_lens = [len(ids) for ids in tokenized_prompts['input_ids']]
    labels = []
    for full_ids, p_len in zip(tokenized_all['input_ids'], prompt_lens):
        label = [-100] * p_len + full_ids[p_len:]
        labels.append(label)
    return {
        'input_ids': tokenized_all['input_ids'],
        'attention_mask': tokenized_all['attention_mask'],
        'labels': labels
    }


def add_length(example: Dict) -> Dict:
    logger.debug("Length col has been added")
    return {'length': len(example['input_ids'])}


def get_cleaned_sorted_dataset(dataset):
    return dataset.sort('length').remove_columns(['instruction', 'output', 'length', 'prompt'])


def split_and_sort_dataset(
        dataset: Dataset,
        config: Dict,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
) -> Dict[str, Dataset]:
    """
    Split dataset into train/val/test, add lengths, clean, and sort each by length.

    Args:
        dataset: Full HF Dataset (tokenized).
        val_ratio: Val proportion
        test_ratio: Test proportion. Proportions must sum to 1.0.
        config: Dict.

    Returns:
        Dict of {'train': Dataset, 'val': Dataset, 'test': Dataset}.
    """
    # First split: train vs test
    temp_rt = val_ratio + test_ratio
    split_1 = dataset.train_test_split(test_size=temp_rt, seed=config["seed"])
    train_ds = split_1["train"]
    temp_ds = split_1["test"]

    # Second split: temp into val vs test
    split_2 = temp_ds.train_test_split(test_size=test_ratio / temp_rt, seed=config["seed"])
    val_ds = split_2["train"]
    test_ds = split_2["test"]

    # Log to check if split is correct
    logger.debug(f"Whole dataset length: {len(dataset)}. "
                 f"Train + val + test length: {len(train_ds) + len(val_ds) + len(test_ds)}")

    # Add length, sort, clean each
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    for name, ds in splits.items():
        # Add length column
        ds = ds.map(add_length)
        # Clean and sort
        ds = get_cleaned_sorted_dataset(ds)
        splits[name] = ds

    return splits


def get_tokenized_dataset(dataset, config):
    # For debugging
    if config["use_small_model"]:
        logger.debug(f"config['use_small_model] = {config['use_small_model']}")
        config['model']['model_name_or_path'] = "gpt2"
        logger.debug(f"Using small model: '{config['model']['model_name_or_path']}'")

    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name_or_path'], padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=100,
        fn_kwargs={"tokenizer": tokenizer})

    return tokenized_dataset, tokenizer


def get_dataloader(
        tokenized_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        seed: int,
        pad_to_multiple_of: Optional[int] = None
) -> DataLoader:
    collator = DataCollatorForCustomPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset=tokenized_dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collator,
        shuffle=False,  # shuffle=False due to sorting
        generator=generator
    )


def unpack_loaders(config):
    # load dataset
    dataset = load_alpaca_data(
        dataset_name=config["dataset_name"],
        data_subset=config["data_subset"],
    )

    # format dataset
    formatted_dataset = dataset.map(format_prompt, batched=True, batch_size=100)

    # tokenize dataset
    tokenized_dataset, tokenizer = get_tokenized_dataset(formatted_dataset, config)

    # Split the tokenized ds into train/val/test_ds
    datasets = split_and_sort_dataset(
        tokenized_dataset,
        config=config,
    )

    train_ds, val_ds, test_ds = datasets.values()

    # debug
    logger.debug(f"Tokenized train dataset sample: {train_ds[0]}\n\n"
                 f"Tokenized val dataset sample: {val_ds[1]}\n\n"
                 f"Tokenized test dataset sample: {test_ds[2]}")

    # Load dataloaders
    train_loader = get_dataloader(
        train_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
    )
    val_loader = get_dataloader(
        val_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
    )
    test_loader = get_dataloader(
        test_ds,
        tokenizer,
        batch_size=config["dataloader"]["batch_size"],
        seed=config["seed"],
    )

    return train_loader, val_loader, test_loader, tokenizer


class DataCollatorForCustomPadding:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Find max length in this batch
        max_length = max(len(example['input_ids']) for example in batch)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * \
                         self.pad_to_multiple_of

        input_ids_padded = []
        attention_mask_padded = []
        labels_padded = []

        for example in batch:
            padding_length = max_length - len(example['input_ids'])

            # Pad input_ids with pad_token_id
            input_ids_padded.append(example['input_ids'] + padding_length * [self.tokenizer.pad_token_id])

            # Pad attention_mask with 0
            attention_mask_padded.append(example['attention_mask'] + [0] * padding_length)

            # Pad labels with -100 (ignored in loss)
            labels_padded.append(example['labels'] + padding_length * [-100])

        for i, el in enumerate(input_ids_padded):
            # print(f"checkpoint {i}")
            if isinstance(el, str):
                print(i, 'STRING', el)

        # Convert to tensors
        return {
            'input_ids': torch.tensor(input_ids_padded),
            'attention_mask': torch.tensor(attention_mask_padded),
            'labels': torch.tensor(labels_padded)
        }