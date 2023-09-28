import numpy as np
import torch
import pickle
import os
from config import device, TRAIN_ORDER
from sklearn.metrics import roc_auc_score, log_loss
from util.common import DatasetName



def initialize_metrics(models_spec: dict, n_experiments: int, n_batches: int):
    """
    Returns a dictionary. For each model name it is (n_exp, n_batch) zeros matrix
    """
    auc, logloss = {}, {}
    for model_name in models_spec:
        auc[model_name] = np.zeros(shape=(n_experiments, n_batches))
        logloss[model_name] = np.zeros(shape=(n_experiments, n_batches))
    return auc, logloss

def update_metrics(models:dict, eval_data: np.ndarray, experiment_idx: int, batch_idx: int,
                   auc: dict, logloss: dict, saving_path:str, dataset_name: DatasetName):
    if dataset_name == DatasetName.coat:
        x_eval = torch.tensor(eval_data[:, :-1], device=device, dtype=torch.float)
    elif dataset_name == DatasetName.yahooR3:
        x_eval = torch.tensor(eval_data[:, :-1], device=device, dtype=torch.long)
    y_eval = eval_data[:, -1].reshape((-1, 1))
    for model_name, _, _ in TRAIN_ORDER:
        if model_name not in models:
            continue
        checkpoint = torch.load(saving_path + model_name + '.pth')
        models[model_name]['architecture'].load_state_dict(checkpoint['model_state_dict'])
        models[model_name]['architecture'].to(device)
        with torch.no_grad():
            if 'dropout' in model_name:
                models[model_name]['architecture'].train()
                preds = torch.zeros((x_eval.shape[0], 1), device=device, dtype=torch.float)
                for i in range(models[model_name]['config']['n_samples']):
                    preds += models[model_name]['architecture'](x_eval)
                preds /= models[model_name]['config']['n_samples']
            else:
                models[model_name]['architecture'].eval()
                preds = models[model_name]['architecture'](x_eval)
            preds = preds.detach().cpu().numpy()
            y_eval = np.asarray(y_eval)
            auc[model_name][experiment_idx][batch_idx] = roc_auc_score(y_eval, preds)
            logloss[model_name][experiment_idx][batch_idx] = log_loss(y_eval, preds)

def get_last_batch_result(auc: dict, logloss: dict):
    last_batch_auc, last_batch_logloss = {}, {}
    for model_name in auc:
        last_batch_auc[model_name] = auc[model_name][:, -1].mean()
        last_batch_logloss[model_name] = logloss[model_name][:, -1].mean()
    return last_batch_auc, last_batch_logloss


def save_results(auc: dict, logloss: dict, experiment_config: dict, data_config: dict):
    path = experiment_config['result_path']
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'auc.pkl', 'wb') as f_handle:
        pickle.dump(auc, f_handle)
    with open(path + 'logloss.pkl', 'wb') as f_handle:
        pickle.dump(logloss, f_handle)
    with open(path + 'experiment_config.txt', 'w') as f_handle:
        print(experiment_config, file=f_handle)
    with open(path + 'data_config.txt', 'w') as f_handle:
        print(data_config, file=f_handle)    
