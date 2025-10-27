import hydra

from source.utils import get_vllm_model
from source.dataset import get_chat_dataset
from vllm import SamplingParams


@hydra.main(version_base="1.3")
def main(config):
    model = get_vllm_model(config)
    tokenizer = model.get_tokenizer()

    dataset = get_chat_dataset(config, tokenizer)
    
    sampling_params = SamplingParams(
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        max_tokens=config.sampling.max_tokens
    )
    outputs = model.generate(dataset["text"], sampling_params)

    dataset["text"] = [out.outputs[0].text for out in outputs]
    dataset.to_csv(f"data/{config.dataset.name}/descriptions/{config.variables.exp_name}.csv", index=False)


if __name__ == "__main__":
    main()