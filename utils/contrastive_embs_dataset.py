from functools import reduce
from operator import iadd
import random
import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
import joblib
from joblib import Parallel, delayed
from ptls.frames.coles.split_strategy import AbsSplit
from typing import List


class ContrastiveEmbsDataset(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.coles.CoLESModule

    Args:
        data: source data with feature dicts
        splitter: object from `ptls.frames.coles.split_strategy`.
            Used to split original sequence into subsequences which are samples from one client.
        col_time: column name with event_time
        n_jobs: number of workers requested by the callers. 
            Passing n_jobs=-1 means requesting all available workers for instance matching the number of
            CPU cores on the worker host(s).
    """

    def __init__(self,
                 data: dict,
                 splitter: AbsSplit,
                 col_time: str = 'event_time',
                 col_embs: str | List[str] = None,
                 n_jobs: int = 1,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.col_embs = col_embs
        self.n_jobs = n_jobs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays)

    def _create_split_subset(self, idx, feature_arrays):
        if not self.col_embs:  # пусто → меток нет
            label = None
        elif isinstance(self.col_embs, str):  # одна строка
            label = feature_arrays[self.col_embs]
            # print(feature_arrays)
            # assert False
        else:  # список с несколькими колонками
            label = [feature_arrays[col] for col in self.col_embs]
        seq_dict = {k: v[idx] for k, v in feature_arrays.items() if self.is_seq_feature(k, v) and k != self.col_embs}
        
        return seq_dict, label

    def get_splits(self, feature_arrays: dict):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        with joblib.parallel_backend(backend='threading', n_jobs=self.n_jobs):
            parallel = Parallel()
            result_dict = parallel(delayed(self._create_split_subset)(idx, feature_arrays)
                                   for idx in indexes)
        return result_dict

    @staticmethod
    def collate_fn(batch):
        batch = reduce(iadd, batch)
        batch, embs = zip(*batch)
        # print(len(embs))
        # print(len(embs[1]))
        embs = torch.vstack(embs).to(torch.float32)
        padded_batch = collate_feature_dict(batch)
        return padded_batch, embs


class ContrastiveEmbsIterableDataset(ContrastiveEmbsDataset, torch.utils.data.IterableDataset):
    pass