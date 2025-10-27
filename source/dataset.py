import pandas as pd
import numpy as np
import torch

from transformers import DataCollatorWithPadding
from tqdm import tqdm
from torch.utils.data import DataLoader


class DataCollatorWithUserIds:
    def __init__(
        self,
        tokenizer,
    ):
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
        )

    def __call__(self, features):
        user_ids = [sample["user_id"] for sample in features]
        features = [sample["inputs"] for sample in features]
        batch = self.data_collator(features)
        batch["user_ids"] = torch.tensor(user_ids)
        return batch


def feature_to_text(features):
    text = [f"{col}: {features[col]}" for col in features.index[1:]]
    return "\n".join(text)


def stats_to_text(config, stat_dataset):
    stats_text = pd.DataFrame({
        config.dataset.col_id: stat_dataset[config.dataset.col_id],
        "text": stat_dataset.apply(feature_to_text, axis=1)
    })
    return stats_text


def build_prompt(config, user_text, tokenizer):
    messages = [
        {"role": "system", "content": config.chat_messages.system},
        {"role": "user", "content": config.chat_messages.user + user_text}
    ]

    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def get_chat_dataset(config, tokenizer):
    stats_text = stats_to_text(
        config, pd.read_csv(config.chat_messages.path_to_input_text)
    )

    stats_text["text"] = stats_text["text"].apply(
        lambda x: build_prompt(config, x, tokenizer)
    )
    return stats_text


def get_inference_dataset(
    config, 
    tokenizer
):
    transactions_text = pd.read_csv(
        config.variables.path_to_inference
    )[[config.dataset.col_id, 'text']]
    
    tokenized_transactions = [
    {
        "user_id": user_id,
        "inputs": tokenizer(
            prompt,   
            max_length=16384,
            truncation=True,
        )} for user_id, prompt in tqdm(transactions_text.values)
    ]

    collator = DataCollatorWithUserIds(tokenizer)

    inference_loader = DataLoader(
        dataset=tokenized_transactions, 
        collate_fn=collator, 
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )
    return inference_loader