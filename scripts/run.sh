export WORK_DIR=$HOME/zoloev-city/gigachat
export WANDB_API_KEY=2736e3a99574e3049342cd33a3154aa307a08aa1
export WANDB_PROJECT="gigachat"
export WANDB_DIR=$WORK_DIR/checkpoints

config_dir=$WORK_DIR/source/llm-foundry/scripts/train/yamls/pretrain
config=hf-llama-3.2-3B.yaml

source $WORK_DIR/source/llm-foundry/llmfoundry-venv/bin/activate


python $WORK_DIR/source/llm4trx/convert_to_text.py \
    --config-dir $config_dir \
    --config-name $config \
    variables.work_dir=$WORK_DIR


python $WORK_DIR/source/llm-foundry/scripts/data_prep/convert_dataset_json.py \
    --config-dir $config_dir \
    --config-name $config \
    variables.work_dir=$WORK_DIR
    

composer $WORK_DIR/source/llm-foundry/scripts/train/train.py \
    $config_dir/$config \
    variables.work_dir=$WORK_DIR


python $WORK_DIR/source/llm-foundry/scripts/inference/convert_composer_to_hf.py \
    --config-dir $config_dir \
    --config-name $config \
    variables.work_dir=$WORK_DIR


accelerate launch $WORK_DIR/source/llm4trx/inference.py \
    --config-dir $config_dir \
    --config-name $config \
    variables.work_dir=$WORK_DIR
