from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_open_thoughts(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = []
        # add system prompt
        messages.append({"role": "system", "content": example['system']})
        # add dialogue
        for message in example['conversations']:
            messages.append({"role": message["from"], "content": message["value"]})
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def prepare_open_platypus(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("garage-bAInd/Open-Platypus", split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = [
            {"role": "user", "content": example["instruction"]}, 
            {"role": "assistant", "content":  example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def prepare_fineweb_edu(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    train_dataset_raw = train_dataset_raw.shuffle(seed=seed, buffer_size=1_000)
    train_dataset = []
    for i, sample in enumerate(train_dataset_raw):
        if i == num_calibration_samples:
            break
        tokenized_sample = tokenizer(
            sample["text"], 
            max_length=max_sequence_length, 
            truncation=True, 
            return_tensors="pt"
        )
        train_dataset.append(tokenized_sample['input_ids'])
    return train_dataset


def prepare_calibration_dataset(
    dataset_name: str, 
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    if dataset_name == "open-thoughts":
        return prepare_open_thoughts(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "open-platypus":
        return prepare_open_platypus(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "fineweb-edu":
        return prepare_fineweb_edu(tokenizer, max_sequence_length, num_calibration_samples, seed)
    else:
        raise ValueError("Unknown dataset")
