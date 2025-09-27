# # -*- coding: utf-8 -*-
# """lora-qlora-qdora.ipynb"""
#
# from huggingface_hub import login
# login("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")
#
# """Import libs | Seed | Set up variables"""
#
# import numpy as np
# import pandas as pd
# import random
# import torch
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler
# import transformers
# from transformers import AutoTokenizer,  AutoModelForCausalLM, DataCollatorForLanguageModeling, DataCollatorWithPadding
# from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
# from typing import Dict, List, Any
#
#
# # # Seed setting
# # def set_seed(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     transformers.set_seed(seed)
# #     # Ensure deterministic behaviour on CUDA
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False
# #     # Warn if non-deterministic algorithms are used
# #     torch.use_deterministic_algorithms(True, warn_only=True)
#
#
# SEED = 17
# set_seed(SEED)
#
# # Setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# batch_size = 4  # do not use batch_size > 24 so dynamic padding is more efficient. 24 is calculated in
#                 # accordance to the instruction + input + output distribution in the yahma/alpaca-cleaned dataset
#
# model_name = "meta-llama/Llama-3.2-3B"
# dataset_name = "yahma/alpaca-cleaned"
#
# # Load and format dataset
# dataset = load_dataset(dataset_name, split="train", streaming=False).remove_columns('input')
#
#
# def format_prompt(batch):
#     prompts = []
#     for inst in batch['instruction']:
#         prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes " \
#                  f"the request.\n\n###Instruction: \n{inst}\n\n###Response: \n"
#         prompts.append(prompt)
#     return {'prompt': prompts}
#
#
# formatted_dataset = dataset.map(format_prompt, batched=True, batch_size=10000)
#
#
# # Tokenize
# # def tokenize(batch, tokenizer):
# #     full_texts = [prompt + output + tokenizer.eos_token for prompt, output in zip(batch['prompt'], batch['output'])]
# #     tokenized_all = tokenizer(full_texts)
# #     tokenized_prompts = tokenizer(batch['prompt'])
# #     prompt_lens = [len(ids) for ids in tokenized_prompts['input_ids']]
# #     labels = []
# #     for full_ids, p_len in zip(tokenized_all['input_ids'], prompt_lens):
# #         label = [-100] * p_len + full_ids[p_len:]
# #         labels.append(label)
# #     return {
# #         'input_ids': tokenized_all['input_ids'],
# #         'attention_mask': tokenized_all['attention_mask'],
# #         'labels': labels
# #     }
#
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# tokenizer.pad_token = tokenizer.eos_token
# tokenized_dataset = formatted_dataset.map(tokenize, batched=True, batch_size=10000,
#                                           fn_kwargs={"tokenizer": tokenizer})
#
#
# # Add sequence length column and sort
# def add_length(example):
#     return {'length': len(example['input_ids'])}
#
#
# tokenized_dataset = tokenized_dataset.map(add_length).sort('length')
# tokenized_dataset = tokenized_dataset.remove_columns(['instruction', 'output', 'length', 'prompt'])
#
#
# class DataCollatorForCustomPadding:
#     def __init__(self, tokenizer, pad_to_multiple_of=None):
#         self.tokenizer = tokenizer
#         self.pad_to_multiple_of = pad_to_multiple_of
#
#     def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
#         # Find max length in this batch
#         max_length = max(len(example['input_ids']) for example in batch)
#         if self.pad_to_multiple_of:
#             max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
#
#         input_ids_padded = []
#         attention_mask_padded = []
#         labels_padded = []
#
#         for example in batch:
#             padding_length = max_length - len(example['input_ids'])
#
#             # Pad input_ids with pad_token_id
#             input_ids_padded.append(example['input_ids'] + padding_length * [self.tokenizer.pad_token_id])
#
#             # Pad attention_mask with 0
#             attention_mask_padded.append(example['attention_mask'] + [0] * padding_length)
#
#             # Pad labels with -100 (ignored in loss)
#             labels_padded.append(example['labels'] + padding_length * [-100])
#
#         for i, el in enumerate(input_ids_padded):
#             # print(f"checkpoint {i}")
#             if isinstance(el, str):
#                 print(i, 'STRING', el)
#
#         # Convert to tensors
#         return {
#             'input_ids': torch.tensor(input_ids_padded),
#             'attention_mask': torch.tensor(attention_mask_padded),
#             'labels': torch.tensor(labels_padded)
#         }
#
#
# collator = DataCollatorForCustomPadding(tokenizer=tokenizer, pad_to_multiple_of=None)
# generator = torch.Generator().manual_seed(SEED)
# train_loader = DataLoader(dataset=tokenized_dataset, batch_size=batch_size, pin_memory=True, collate_fn=collator, shuffle=False, generator=generator)  # shuffle=False due to sorting
#
# """Training"""
#
# model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, dtype=torch.float16,
#                                              offload_folder="offload")
#
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=8, use_rslora=True, target_modules=['q_proj', 'v_proj'])
#
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
#
# lr = 1e-4
# num_epochs = 3
# grad_accumulation_steps = 4
# num_training_steps = num_epochs * (len(train_loader) // grad_accumulation_steps)
# num_warmup_steps = int(0.03 * num_training_steps)
#
# scaler = GradScaler()
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8)
# scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
#
# model.to(device)
#
#
# for epoch in range(1, num_epochs+1):
#     train_losses = []
#     log_every = 500
#     running_loss = 0
#
#     print(f"\n\n<<<@@@### EPOCH {epoch} ###@@@>>>\n\n")
#
#     model.train()
#     for step, batch in enumerate(train_loader):
#         input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
#         with autocast(dtype=torch.float16):
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss / grad_accumulation_steps
#
#         # Backward pass with scaled loss
#         scaler.scale(loss).backward()
#
#         running_loss += loss.item()
#
#         # gradient accumulation
#         if step % grad_accumulation_steps == 0:
#             # clip grads *after* scaling
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#             optimizer.zero_grad()
#
#         if (step+1) % log_every == 0:
#             print(f"Step {step+1}: average loss = {running_loss / log_every}")
#             running_loss = 0
#
#     # check for a leftover
#     if (step + 1) % grad_accumulation_steps != 0:
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()
#         optimizer.zero_grad()
#
#     epoch_loss = sum(train_losses) / len(train_losses)
#     print(f"\n\nEpoch {epoch} average loss: {epoch_loss}\n\n")
#
#     # release unused cached memory back to the GPU
#     torch.cuda.empty_cache()
#
