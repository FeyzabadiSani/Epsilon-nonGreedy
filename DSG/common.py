import numpy as np
import torch
from copy import deepcopy
from util.common import DataType, DatasetName
from config import device
from DSG.coatDSG import get_cartesian_product as coat_cartesian_product
from DSG.r3DSG import get_cartesian_product as r3_cartesian_product

def cartesian_product(data: torch.Tensor, dataset_name: DatasetName):
    if dataset_name == DatasetName.coat:
        return coat_cartesian_product(data)
    elif dataset_name == DatasetName.yahooR3:
        return r3_cartesian_product(data)
    raise Exception("Not Valid Dataset")


def initialize_models_training_dataset(models_spec: dict):
    """
    Retuerns a dictionary conrains None for each model
    """
    datasets = {}
    for model_name in models_spec:
        # Autodebias Net needs uniform dataset too
        if model_name == 'autodebias-net':
            datasets['autodebias-uniform'] = None

        datasets[model_name] = None
    return datasets


def select_winners(model: torch.nn.Module, pick_ratio: float,
                   biased_batch: np.ndarray, dataset_name: DatasetName):
    model.to(device)
    if dataset_name == DatasetName.coat:
        data = torch.tensor(biased_batch[:, :-1], device=device, dtype=torch.float)
    elif dataset_name == DatasetName.yahooR3:
        data = torch.tensor(biased_batch[:, :-1], device=device, dtype=torch.long)
    preds = model(data)
    preds = torch.reshape(preds, (-1, ))
    idx = torch.argsort(preds, descending=True).detach().cpu().numpy()
    idx = idx[:int(idx.shape[0] * pick_ratio)]
    return biased_batch[idx]


def update_datasets(models: dict, datasets: dict, path: str, pick_ratio: float,
                    uniform_batch: np.ndarray, biased_batch: np.ndarray,
                    dataset_name: DatasetName):                       
    for model_name in models:
        # Autodebias Net needs uniform dataset too
        if model_name == 'autodebias-net':
            if datasets['autodebias-uniform'] is None:
                datasets['autodebias-uniform'] = uniform_batch
            else:
                datasets['autodebias-uniform'] = np.concatenate((datasets[model_name], uniform_batch), axis=0)
                
        if models[model_name]['config']['data_type'] == DataType.uniform:
            if datasets[model_name] is None:
                datasets[model_name] = uniform_batch
            else:
                datasets[model_name] = np.concatenate((datasets[model_name], uniform_batch), axis=0)
        elif models[model_name]['config']['data_type'] == DataType.biased:
            checkpoint = deepcopy(models[model_name]['architecture'])
            if datasets[model_name] is not None:
                checkpoint.load_state_dict(torch.load(path+model_name+'.pth')['model_state_dict'])
                if 'dropout' in model_name:
                    checkpoint.train()
                else:
                    checkpoint.eval()
            
            update_batch = select_winners(
                model=checkpoint,
                pick_ratio=pick_ratio,
                biased_batch=biased_batch,
                dataset_name=dataset_name,
            )
            if datasets[model_name] is not None:
                datasets[model_name] = np.concatenate((datasets[model_name], update_batch), axis=0)
            else:
                datasets[model_name] = update_batch
        else:
            checkpoint = deepcopy(models[model_name]['architecture'])
            if datasets[model_name] is not None:
                checkpoint.load_state_dict(torch.load(path+model_name+'.pth')['model_state_dict'])
                if 'dropout' in model_name:
                    checkpoint.train()
                else:
                    checkpoint.eval()
            update_batch = select_winners(
                model=checkpoint,
                pick_ratio=pick_ratio,
                biased_batch=biased_batch,
                dataset_name=dataset_name,
            )
            if datasets[model_name] is not None:
                datasets[model_name] = np.concatenate((datasets[model_name], uniform_batch, update_batch), axis=0)
            else:
                datasets[model_name] = np.concatenate((uniform_batch, update_batch), axis=0)