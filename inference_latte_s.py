import hydra
import torch
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
import os

from ptls.data_load.utils import collate_feature_dict
from congpt.utils.ptls_extensions import InferenceModule
import logging

logger = logging.getLogger(__name__)

def save_scores(df_scores, output_conf):
    output_name = output_conf.path
    output_format = output_conf.format
    if output_format not in ('pickle', 'csv', 'parquet'):
        logger.warning(f'Format "{output_format}" is not supported. Used default "pickle"')
        output_format = 'pickle'

    output_path = f'{output_name}.{output_format}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_format == 'pickle':
        df_scores.to_pickle(output_path)
    elif output_format == 'csv':
        df_scores.to_csv(output_path, sep=',', header=True, index=False)
    elif output_format == 'parquet':
        df_scores.to_parquet(output_path)
    logger.info(f'{len(df_scores)} records saved to: "{output_path}"')


def inference_module(conf: DictConfig):
    method = conf['method']
    model = hydra.utils.instantiate(conf[method].pl_module)
    model = model.seq_encoder
    model.is_reduce_sequence = True
    model.reducer = conf.inference.get('pooling_strategy', 'last_step')
    if conf.get('finetune', False):
        model.load_state_dict(torch.load(conf.finetuned_model_path))
    else:
        model.load_state_dict(torch.load(conf.model_path))
    
    model = InferenceModule(
        model=model,
        pandas_output=True, model_out_name='emb',
    )

    dataset_inference = hydra.utils.instantiate(conf.inference.dataset)
    inference_dl = DataLoader(
        dataset=dataset_inference,
        collate_fn=collate_feature_dict,
        shuffle=False,
        num_workers=conf.inference.get('num_workers', 8),
        batch_size=conf.inference.get('batch_size', 128),
    )
    devices = conf.inference.get('devices', 1)

    df_scores = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=-1).predict(model, inference_dl)
    df_scores = pd.concat(df_scores, axis=0)
    logger.info(f'df_scores examples: {df_scores.shape}:')
    
    save_scores(df_scores, conf.inference.output)
    return df_scores


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    return inference_module(conf)

if __name__ == '__main__':
    main()