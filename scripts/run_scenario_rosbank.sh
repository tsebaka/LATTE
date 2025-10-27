#!/bin/bash


CONFIG=rosbank_descriptions.yaml

python generate_descriptions.py \
    --config-dir configs \
    --config-name $CONFIG

accelerate launch inference.py \
    --config-dir configs \
    --config-name $CONFIG
