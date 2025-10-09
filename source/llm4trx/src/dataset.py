import peft
import pandas as pd
import numpy as np
import torch

from src.utils.utils import DataCollatorWithUserIds, get_feature_preprocessor
from tqdm import tqdm
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

tqdm.pandas(leave=True)


def trx_to_text_converter(
    config,
    transaction,
    preprocessor=None,
    tokenizer=None,
    chat=False,
):
    header = config.variables.dataset.header_separator.join(config.variables.dataset.header_features)

    transactions = [header] + [
        config.variables.dataset.feature_separator.join(
            preprocessor.preprocess(
                config, 
                transaction[feature][timestamp], 
                feature
            ) for feature in list(config.variables.dataset.features)
        ) for timestamp in range(len(transaction[config.variables.dataset.features[0]]))
    ]
    text = config.variables.dataset.trx_separator.join(transactions)

    if chat:
        messages = [
            {"role": "system", "content": config.variables.dataset.chat_messages.system},
            {"role": "user", "content": config.variables.dataset.chat_messages.user + text}
        ]

        text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )

    return int(transaction[config.variables.dataset.col_id]), text


def convertation(
    config,
    tokenizer
):
    transactions = pd.read_parquet(config.variables.dataset.train_path)
    preprocessor = get_feature_preprocessor(config)
    
    transactions_text = transactions.progress_apply(
        lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )
    transactions_text = pd.DataFrame(list(transactions_text), columns=["user_id", "text"])

    transactions_text.to_json(
        config.variables.text_convertation.out_file,
        orient="records", 
        lines=True
    )
    return transactions_text


def get_inference_dataset(
    config, 
    tokenizer
):
    transactions = pd.concat((
        pd.read_parquet(config.variables.dataset.train_path),
        pd.read_parquet(config.variables.dataset.test_path))
    ).reset_index(drop=True)

    preprocessor = get_feature_preprocessor(config)
    
    transactions_text = transactions.progress_apply(
        lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )
    transactions_text = pd.DataFrame(list(transactions_text), columns=["user_id", "inputs"])
    tokenized_transactions = [
    {
        "user_id": user_id,
        "inputs": tokenizer(
            prompt,   
            max_length=16384, # подумать
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
