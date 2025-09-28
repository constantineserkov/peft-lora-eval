import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Optional, Any
from src.logger import get_logger

logger = get_logger()


def load_alpaca_data(dataset_name: str, data_subset: int) -> Dataset:
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
    return {'length': len(example['input_ids'])}


def get_cleaned_sorted_dataset(dataset):
    return dataset.sort('length').remove_columns(['instruction', 'output', 'length', 'prompt'])


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