#!/usr/bin/env bash

export PYTHONPATH="../../"
SPARK_LOCAL_IP="127.0.0.1" spark-submit \
    --master local[8] \
    --name "Age Make Dataset" \
    --driver-memory 16G \
    --conf spark.sql.shuffle.partitions=100 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    make_dataset.py \
    --data_path data/ \
    --trx_files transactions_train.csv transactions_test.csv \
    --col_client_id "client_id" \
    --cols_event_time "#float" "trans_date" \
    --cols_category "trans_date" "small_group" \
    --cols_log_norm "amount_rur" \
    --target_files train_target.csv \
    --save_partitioned_data \
    --col_target bins \
    --test_size 0.1 \
    --output_train_path "data/train_trx_part.parquet" \
    --output_test_path "data/test_trx_part.parquet" \
    --output_test_ids_path "data/test_ids_part.csv" \
    --log_file "results/dataset_age_pred_part.txt"

# 654 sec with    --print_dataset_info
# 144 sec without --print_dataset_info
