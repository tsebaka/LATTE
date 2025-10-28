#!/bin/bash
CONFIG_NAME=con_embs
METHOD=con_embs
echo "Starting training..."
echo "Starting pretraining..."
python -m congpt.train --config-dir conf --config-name params_rnn finetune=false +method=coles

echo "Starting inference..."
python -m congpt.inference --config-dir conf --config-name $CONFIG_NAME finetune=false +method=$METHOD

echo "Starting validation..."
python -m congpt.validation --config-dir conf --config-name $CONFIG_NAME