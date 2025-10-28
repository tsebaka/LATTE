import torch
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_step import LastStepEncoder
from congpt.utils.ptls_extensions import AvgEncoder
from ptls.data_load.padded_batch import PaddedBatch

class SeqEncoder(torch.nn.Module):
    def __init__(self, trx_encoder: TrxEncoder, 
                 seq_encoder: torch.nn.Module,
                 is_reduce_sequence: bool = True,
                 reducer: str = 'last_step'):
        super().__init__()
        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder
        self._is_reduce_sequence = is_reduce_sequence
        self._reducer = {
            'last_step': LastStepEncoder(),
            'avg': AvgEncoder(),
        }[reducer]
        
    def forward(self, x, *args, **kwargs):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x)
        if self._is_reduce_sequence:
            x = self.reducer(x)
        return x
    
    @property
    def is_reduce_sequence(self):
        return self._is_reduce_sequence
    
    @is_reduce_sequence.setter
    def is_reduce_sequence(self, is_reduce_sequence: bool):
        self._is_reduce_sequence = is_reduce_sequence

    @property
    def reducer(self):
        return self._reducer
    
    @reducer.setter
    def reducer(self, reducer: str):
        self._reducer = {
            'last_step': LastStepEncoder(),
            'avg': AvgEncoder(),
        }[reducer]

# class VQSeqEncoder(torch.nn.Module):
#     def __init__(self, vq_encoder, seq_encoder, is_reduce_sequence=False, reducer='last_step', return_indices=True):
#         super().__init__()
#         self.vq_encoder = vq_encoder
#         self.indices_encoder = TrxEncoder(
#             embeddings={
#                 "indices": {"in": 4096, "out": 512},
#             },
#         )
#         self.seq_encoder = seq_encoder
#         self._is_reduce_sequence = is_reduce_sequence
#         self._reducer = {
#             'last_step': LastStepEncoder(),
#             'avg': AvgEncoder(),
#         }[reducer]
#         self.embedding_size = seq_encoder.n_embd
#         self.return_indices = return_indices

#     def forward(self, x, *args, **kwargs):
#         indices_dict = {}
#         # print(x.payload['amount_rur'].size())
#         with torch.no_grad():
#             seq_lens = x.seq_lens
#             x = self.vq_encoder(x)['indices']
#             # print(x)
#             indices_dict['indices'] = x
#             indices_dict['event_time'] = torch.arange(x.shape[1], device=indices_dict['indices'].device).repeat(x.shape[0], 1) 
#             # print(indices_dict) 
#         indices = PaddedBatch(indices_dict, seq_lens)
#         # print(x.payload['indices'].size())
#         x = self.indices_encoder(indices)
#         x = self.seq_encoder(x)
#         # print(x.payload)
#         if self._is_reduce_sequence:
#             x = self.reducer(x)
#         # print(x.payload)
#         if self.return_indices:
#             return x, indices
#         else:
#             return x
    
#     @property
#     def is_reduce_sequence(self):
#         return self._is_reduce_sequence
    
#     @is_reduce_sequence.setter
#     def is_reduce_sequence(self, is_reduce_sequence: bool):
#         self._is_reduce_sequence = is_reduce_sequence

#     @property
#     def reducer(self):
#         return self._reducer
    
#     @reducer.setter
#     def reducer(self, reducer: str):
#         self._reducer = {
#             'last_step': LastStepEncoder(),
#             'avg': AvgEncoder(),
#         }[reducer]