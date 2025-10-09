for SC_AMOUNT in 290000 200000 100000 050000 025000 012000 006000 003000 001000 000500
do
      python -m ptls.pl_fit_target \
        logger_name="fit_target_${SC_AMOUNT}" \
        data_module.train.labeled_amount=$SC_AMOUNT \
        embedding_validation_results.feature_name="target_scores_${SC_AMOUNT}" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/fit_target_${SC_AMOUNT}_results.json" \
        --config-dir conf --config-name pl_fit_target_rnn

    python -m ptls.pl_fit_target \
        logger_name="mles_finetuning_${SC_AMOUNT}" \
        data_module.train.labeled_amount=$SC_AMOUNT \
        embedding_validation_results.feature_name="mles_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --config-dir conf --config-name pl_fit_finetuning_on_mles

    python -m ptls.pl_fit_target \
        logger_name="cpc_finetuning_${SC_AMOUNT}" \
        data_module.train.labeled_amount=$SC_AMOUNT \
        embedding_validation_results.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --config-dir conf --config-name pl_fit_finetuning_on_cpc
done

rm results/scenario_x5__semi_supervised.txt
# rm -r conf/embeddings_validation_semi_supervised.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_semi_supervised +workers=10 +total_cpu_count=20
