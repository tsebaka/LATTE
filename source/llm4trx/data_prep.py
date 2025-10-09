import warnings
import hydra
import numpy as np
import omegaconf
import torch

from omegaconf import OmegaConf
from src.dataset.dataset import get_train_dataset
from src.utils.utils import (
    get_model,
    get_tokenizer,
    get_data_collator
)
from transformers import (
    set_seed,
)

warnings.filterwarnings("ignore")


def set_global_seed(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)
    set_seed(config.seed)


def data_prep(
    config,
):
    transactions_train = pd.read_parquet(config.dataset.train_path)
    transactions_test = pd.read_parquet(config.dataset.test_path)

    transactions = pd.concat((
        transactions_train,
        transactions_test)
    ).reset_index(drop=True)

    transactions_text = transactions.progress_apply(
        lambda row: trx_to_text_converter(
            config, 
            row, 
            preprocessor=preprocessor, 
            inference=True), 
        axis=1
    )
    transactions_text = pd.DataFrame(list(transactions_text), columns=["user_id", "inputs"])
    

@hydra.main(version_base=None, config_path="config")
def main(config):
    set_global_seed(config)
    data_prep(config)


if __name__ == "__main__":
    main()