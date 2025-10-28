import pytorch_lightning as pl
from torch.nn import Module
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from functools import partial
import torch
from typing import Optional, Iterable, Dict
from congpt.utils.losses import InfoNCELoss

class ContrastiveEmbsModule(pl.LightningModule):
    def __init__(self,
                 seq_encoder: SeqEncoderContainer,
                 rnn_head: Module,
                 embs_head: Optional[Module] = None,
                 pretrained_model_path:  Optional[str] = None,
                 loss: Module = InfoNCELoss(),
                 seq_encoder_lr: Optional[float] = None,
                 embs_head_lr: Optional[float] = None,
                 rnn_head_lr: Optional[float] = None,
                 optimizer_partial: Optional[partial] = None,
                 lr_scheduler_partial: Optional[partial] = None,
                 weight_decay: float = 0.0):
        super().__init__()
        self._seq_encoder = seq_encoder
        if pretrained_model_path is not None:
            pretrained_model = torch.load(pretrained_model_path)
            self._seq_encoder.load_state_dict(pretrained_model)
        self._seq_encoder.eval()
        for param in self._seq_encoder.parameters():
            param.requires_grad = False
        self._embs_head = embs_head
        self._rnn_head = rnn_head
        self._loss = loss

        # сохраним лернинги
        self._seq_encoder_lr = seq_encoder_lr
        self._embs_head_lr = embs_head_lr
        self._rnn_head_lr = rnn_head_lr

        # оптимизатор/шедулер-«фабрики»
        self._optimizer_partial = optimizer_partial or partial(torch.optim.AdamW, lr=1e-3)
        self._lr_scheduler_partial = lr_scheduler_partial
        self._weight_decay = weight_decay

        # удобно логировать гиперпараметры
        self.save_hyperparameters(ignore=['seq_encoder', 'embs_head', 'rnn_head', 'loss'])

    def forward(self, x, embs):
        rnn_embs = self._rnn_head(self._seq_encoder(x))
        if self._embs_head is not None:
            embs = self._embs_head(embs)
        return embs, rnn_embs

    def training_step(self, batch, batch_idx):
        x, embs = batch
        z_e, z_rnn = self(x, embs)
        loss = self._loss(z_e, z_rnn)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, embs = batch
        z_e, z_rnn = self(x, embs)
        loss = self._loss(z_e, z_rnn)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _group(self, module: Optional[Module], lr: Optional[float]) -> Optional[Dict]:
        if module is None:
            return None
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return None
        group = {'params': params}
        if lr is not None:
            group['lr'] = lr
        if self._weight_decay:
            group['weight_decay'] = self._weight_decay
        return group

    def configure_optimizers(self):
        # собираем группы параметров с разными lr
        param_groups = []

        g_seq = self._group(self._seq_encoder, self._seq_encoder_lr)
        if g_seq is not None:
            param_groups.append(g_seq)

        g_emb = self._group(self._embs_head, self._embs_head_lr)
        if g_emb is not None:
            param_groups.append(g_emb)

        g_rnn = self._group(self._rnn_head, self._rnn_head_lr)
        if g_rnn is not None:
            param_groups.append(g_rnn)

        # если ни одной спец-группы не вышло — один дефолтный пул
        if not param_groups:
            param_groups = [{'params': [p for p in self.parameters() if p.requires_grad]}]

        optimizer = self._optimizer_partial(param_groups)

        # шедулер опционален
        if self._lr_scheduler_partial is None:
            return optimizer

        scheduler = self._lr_scheduler_partial(optimizer)

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

    @property
    def seq_encoder(self):
        return ContrastiveEmbsInferenceModule(self)

class ContrastiveEmbsInferenceModule(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self._model = pretrained_model
        

    def forward(self, batch):
        rnn_embs = self._model._seq_encoder(batch)
        head_rnn = self._model._rnn_head(rnn_embs)
        # return head_rnn
        return torch.cat([rnn_embs, head_rnn], dim=1)