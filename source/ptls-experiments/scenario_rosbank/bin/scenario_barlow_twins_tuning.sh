export SC_SUFFIX="bt_tuning_base"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_hidden_size_0680"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=680 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_batch_size_96"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=96 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_lambd_0.02"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
    --config-dir conf --config-name barlow_twins_params


#################

export SC_SUFFIX="bt_tuning_v01"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256}], [ReLU, {}], [Linear, {in_features: 256, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v02"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128}], [ReLU, {}], [Linear, {in_features: 128, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v03"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512}], [ReLU, {}], [Linear, {in_features: 512, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0.001 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v04"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=2048 \
    "params.head_layers=[[Linear, {in_features: 2048, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512}], [ReLU, {}], [Linear, {in_features: 512, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512, affine: False}]]" \
    data_module.train.batch_size=256 \
    params.train.lr=0.002 \
    params.train.weight_decay=0.001 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=400 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v05"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.01 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256}], [ReLU, {}], [Linear, {in_features: 256, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.7 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v06"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128}], [ReLU, {}], [Linear, {in_features: 128, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.7 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v07"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[BatchNorm1d, {num_features: 1024}], [ReLU, {}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v08"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Dropout, {p: 0.1}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v09"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.08 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[ReLU, {}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v10"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.002 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[ReLU, {}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params


export SC_SUFFIX="bt_tuning_v11"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v12"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v13"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.002 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params


export SC_SUFFIX="bt_tuning_v14"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.batch_size=1024 \
    params.train.lambd=0.04 \
    params.train.lr=0.004 \
    params.lr_scheduler.step_size=10 \
    trainer.max_epochs=600 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v15"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.batch_size=1024 \
    params.train.lambd=0.04 \
    params.train.lr=0.004 \
    params.lr_scheduler.step_size=50 \
    trainer.max_epochs=600 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v16"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.batch_size=1024 \
    params.train.lambd=0.02 \
    params.train.lr=0.004 \
    params.lr_scheduler.step_size=50 \
    trainer.max_epochs=600 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v18"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.batch_size=512 \
    params.train.lambd=0.04 \
    params.train.lr=0.004 \
    params.lr_scheduler.step_size=40 \
    trainer.max_epochs=600 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params


export SC_SUFFIX="bt_tuning_v17"; export SC_VERSION=1
export SC_SUFFIX="bt_tuning_v15"; export SC_VERSION=0
export SC_SUFFIX="bt_tuning_v16"; export SC_VERSION=0

ls "lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/"
# ep = 9; st = 79; {i: (st + 1) // (ep + 1) * (i + 1) - 1 for i in range(ep, 600, 10)}

python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=9-step\=79.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_009" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=49-step\=399.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_049" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=99-step\=799.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_099" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=149-step\=1199.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_149" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=199-step\=1599.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_199" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=249-step\=1999.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_249" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=299-step\=2399.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_299" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=349-step\=2799.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_349" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=399-step\=3199.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_399" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=449-step\=3599.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_449" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=200 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=499-step\=3999.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_499" \
    --config-dir conf --config-name barlow_twins_params


rm results/res_bt_tuning.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/res_bt_tuning.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__bt_tuning_*.pickle", "${hydra:runtime.cwd}/data/barlow_twins_embeddings.pickle"]
less -S results/res_bt_tuning.txt

