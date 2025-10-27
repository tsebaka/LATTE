import torch

from vllm import LLM
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModel
)


def get_vllm_model(
    config: DictConfig
):
    model = LLM(
        model=config.models.description_model.path,
        task='generate',
        dtype=torch.bfloat16,
        seed=config.seed,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95
    )
    return model


def get_inference_model(
    config: DictConfig,
):
    model = AutoModel.from_pretrained(
        config.models.embedding_model.path,
        attn_implementation='flash_attention_2',
        dtype=torch.bfloat16
    )
    return model


def get_tokenizer(
    config: DictConfig,
):
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.embedding_model.path,
    )
    return tokenizer


def get_embedding(
    config,
    batch, 
    model
):
    outputs = model(**batch, output_hidden_states=True)
    
    hidden_states = torch.stack(
        outputs.hidden_states[config.models.embedding_model.pooling:]
    ).mean(dim=0)
    
    attention_mask = batch["attention_mask"].unsqueeze(2)
    embeddings = (
        hidden_states * attention_mask
    ).sum(dim=1) / attention_mask.sum(dim=1)
    
    return embeddings