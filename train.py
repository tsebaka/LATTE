from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
import logging
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os
from lightning_utilities.core.rank_zero import rank_zero_only 
logger = logging.getLogger(__name__)

def build_callbacks(trainer_cfg: dict) -> list:
    callbacks = []
    
    # Checkpoint every N epochs
    every_n = trainer_cfg.pop('checkpoints_every_n_val_epochs', None)
    if every_n:
        cb = ModelCheckpoint(every_n_epochs=every_n, save_top_k=-1)
        logger.info(f"Adding ModelCheckpoint every {every_n} epochs")
        callbacks.append(cb)
    
    # Learning rate monitor
    if trainer_cfg.pop('use_lr_monitor', True):
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    return callbacks

def get_model_and_dm(conf: DictConfig):
    finetune = conf.get('finetune', False)
    method = conf.get('method', 'gpt')
    model = hydra.utils.instantiate(conf[method].pl_module)
    # if method == 'gpt':
    #     model.seq_encoder.seq_encoder.causal = True
    
    if finetune:
        model.seq_encoder.load_state_dict(torch.load(conf.model_path))
        # model.seq_encoder.seq_encoder.causal = conf[method].pl_module.seq_encoder.seq_encoder.get('causal', False)
    dm = hydra.utils.instantiate(conf[method].data_module)
    return model, dm
@rank_zero_only
def save_model(model, conf: DictConfig, finetune: bool = False):
    os.makedirs(os.path.dirname(conf.model_path), exist_ok=True)
    if finetune:
        torch.save(model.seq_encoder.state_dict(), conf.finetuned_model_path)
    else:
        torch.save(model.seq_encoder.state_dict(), conf.model_path)
    logger.info(f'Model weights saved to "{conf.model_path}"')

def train_module(conf: DictConfig) -> None:
    """
    Train Lightning module according to Hydra config.
    """
    # Seed
    if 'seed' in conf:
        pl.seed_everything(conf.seed)
        logger.debug(f"Seed set to {conf.seed}")
    
    model, dm = get_model_and_dm(conf)

    # Prepare trainer params
    if conf.get('finetune', False) and 'finetune_trainer' in conf:
        trainer_cfg = OmegaConf.to_container(conf.finetune_trainer, resolve=True)
    else:
        trainer_cfg = OmegaConf.to_container(conf.trainer, resolve=True)

    callbacks = build_callbacks(trainer_cfg)
    if 'logger_name' in conf:
        trainer_cfg['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )
    if callbacks:
        trainer_cfg['callbacks'] = callbacks

    # Create and run trainer
    trainer = pl.Trainer(**trainer_cfg)
    # Создаем папку для сохранения модели если её нет
    os.makedirs(os.path.dirname(conf.model_path), exist_ok=True)
    if conf.get('finetune', False):
        os.makedirs(os.path.dirname(conf.finetuned_model_path), exist_ok=True)
    trainer.fit(model, dm)

    
    if conf.get('finetune', False):
        # torch.save(model.seq_encoder.state_dict(), conf.finetuned_model_path)
        save_model(model, conf, finetune=True)
        logger.info(f'Model weights saved to "{conf.finetuned_model_path}"')
    else:
        # torch.save(model.seq_encoder.state_dict(), conf.model_path)
        save_model(model, conf, finetune=False)
        logger.info(f'Model weights saved to "{conf.model_path}"')
    return model


@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    return train_module(conf)


if __name__ == '__main__':
    main()