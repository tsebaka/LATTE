import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
import hydra
import gc


def save_results(conf, metrics_valid, metrics_test):
    valid_mean = np.mean(metrics_valid)
    valid_std = np.std(metrics_valid)
    test_mean = np.mean(metrics_test)
    test_std = np.std(metrics_test)
    res_dir = conf.get('res_dir', 'results')
    if (os.path.exists(f'{res_dir}/results.csv')):
        f = open(f'{res_dir}/results.csv','a')
        f.write(f'{conf.logger_name},{valid_mean},{valid_std},{test_mean},{test_std}\n') 
        f.close()
    else:
        os.mkdir(res_dir)
        f = open(f'{res_dir}/results.csv','w')
        f.write('name,valid_mean,valid_std,test_mean,test_std\n') 
        f.write(f'{conf.logger_name},{valid_mean},{valid_std},{test_mean},{test_std}\n') 
        f.close()

def validation(conf):
    df_scores = pd.read_pickle(conf.inference.output.path + '.pickle')
    target_conf = conf.validation.target
    target = pd.read_csv(target_conf.file_name, usecols = [target_conf.cols_id, target_conf.col_target])
    test_ids = pd.read_csv(conf.validation.test_id_file)
    target_train, target_test = target.loc[~target[target_conf.cols_id].isin(test_ids[target_conf.cols_id])], target.loc[target[target_conf.cols_id].isin(test_ids[target_conf.cols_id])]
    df_scores[target_conf.cols_id] = df_scores[target_conf.cols_id].astype('int')
    df_train, df_test = df_scores.merge(target_train, on = target_conf.cols_id), df_scores.merge(target_test, on = target_conf.cols_id)
    del df_scores, target_train, target_test, target
    gc.collect()
    df_train, df_test = df_train.drop(target_conf.cols_id, axis = 1), df_test.drop(target_conf.cols_id, axis = 1)
    X_train, X_test, y_train, y_test = df_train.drop(target_conf.col_target, axis = 1), df_test.drop(target_conf.col_target, axis = 1), df_train[target_conf.col_target], df_test[target_conf.col_target]
    del df_train, df_test
    gc.collect()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if 'preprocessing' in conf.validation:
        for scaler in conf.validation.preprocessing:
            real_scaler = hydra.utils.instantiate(scaler)
            transformer = real_scaler.fit(X_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)
    metric = hydra.utils.instantiate(conf.validation.metric)
    metrics_valid, metrics_test = [], []

    for train, valid in skf.split(X_train, y_train):
        model = hydra.utils.instantiate(conf.validation.model)
        model.fit(X_train[train], y_train.to_numpy()[train])
        if conf.validation.metric_name == 'roc_auc':
            y_pred = model.predict_proba(X_train[valid])[:, 1]
        else:
            y_pred = model.predict(X_train[valid])
        metrics_valid.append(metric(y_train.to_numpy()[valid], y_pred))
        if conf.validation.metric_name == 'roc_auc':
            y_pred_test = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_test = model.predict(X_test)
        metrics_test.append(metric(y_test.to_numpy(), y_pred_test))
    save_results(conf, metrics_valid, metrics_test)
    return np.mean(metrics_valid)

@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    return validation(conf)


if __name__ == '__main__':
    main()