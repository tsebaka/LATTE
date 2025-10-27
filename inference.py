import accelerate
import numpy as np
import pandas as pd
import hydra
import torch
import warnings 

from source.utils import (
    get_tokenizer, 
    get_inference_model, 
    get_embedding,
)
from source.dataset import (
    get_inference_dataset,
    get_chat_dataset
)
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

warnings.filterwarnings('ignore')


def inference(
    config,
):
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    
    tokenizer = get_tokenizer(config)
    model = get_inference_model(config)
    model.eval()

    inference_loader = get_inference_dataset(config, tokenizer)
    model, inference_loader = accelerator.prepare(model, inference_loader)

    with torch.inference_mode(): 
        user_ids = []
        embeddings = []
        for batch in tqdm(inference_loader):
            user_ids.extend(batch.pop("user_ids"))
            embeddings.extend(get_embedding(config, batch, model))

    user_ids_gather = gather_object(user_ids)
    embeddings_gather = gather_object(embeddings)

    if accelerator.is_main_process:
        user_ids = torch.stack([user_id.cpu() for user_id in user_ids_gather]).numpy()
        embeddings = torch.stack([embedding.to(torch.float16).cpu() for embedding in embeddings_gather]).tolist()
        
        embeddings_train_test = pd.DataFrame({
            config.dataset.col_id: user_ids,
            "embedding": embeddings
        })
        for i in tqdm(range(len(embeddings[0]))):
            embeddings_train_test[f'emb_{i+1:04d}'] = embeddings_train_test["embedding"].apply(lambda x: x[i])
            
        embeddings_train_test = (
            embeddings_train_test
            .drop("embedding", axis=1)
            .drop_duplicates(subset=config.dataset.col_id, keep='last')
            .reset_index(drop=True)
        )
        embeddings_train_test.to_parquet(
            config.variables.embeddings_save_path
        )


@hydra.main(version_base=None)
def main(config):
    inference(config)


if __name__ == '__main__':
    main()