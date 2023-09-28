import logging
import pandas as pd
import numpy as np
import torch

from DSG.util import train_test_split, divide_train_data_into_batches

global_cutoff = None
biased_matrix = np.zeros(shape=(15400, 1000), dtype=np.uint8)

def load_dataset(path:str):
    global biased_matrix
    dataset = dict()
    dataset['biased'] = pd.read_csv(
        path + 'ydata-ymusic-rating-study-v1_0-train.txt',
        sep="\t",
        header=None,
        engine="python"
        ).values
    dataset['uniform'] = pd.read_csv(
        path + 'ydata-ymusic-rating-study-v1_0-test.txt', 
        sep="\t", 
        header=None, 
        engine="python"
        ).values
    
    rows = dataset['biased'][:, 0] - 1
    cols = dataset['biased'][:, 1] - 1
    vals = dataset['biased'][:, 2]
    biased_matrix[rows, cols] = vals
    return dataset


def encode_data(data: np.ndarray, cutoff: int):
    user_ids = data[:, 0] - 1
    item_ids = data[:, 1] - 1
    labels = np.where(data[:, 2] > cutoff, 1.0, 0.0)
    user_ids = user_ids.reshape(-1, 1)
    item_ids = item_ids.reshape(-1, 1)
    labels = labels.reshape(-1, 1)
    data = np.concatenate((user_ids, item_ids, labels), axis=1)
    return data


def prepare_data(dataset: dict, cutoff: int):
    data = dict()
    data['st'] = encode_data(
        data=dataset['uniform'],
        cutoff=cutoff,
    )
    data['sc'] = encode_data(
        data=dataset['biased'],
        cutoff=cutoff,
    )
    return data

def log_data(data:dict, logger:logging.Logger):
    """
    Logging prepared data attributes
    """
    logger.error(r"Dataset Statistics")
    logger.error(r"    *Train uniform data shape=%s" % (str(data['st_train'].shape)))
    logger.error(r"    *Train biased data shape=%s" % (str(data['sc'].shape)))
    logger.error(r"    *Validation data shape=%s" % (str(data['st_validation'].shape)))
    logger.error(r"    *Test data shape=%s" % (str(data['st_test'].shape)))
    logger.error(r"    *Number of training batches=%d" % len(data['train']['uniform']))
    logger.error(r"    *Uniform train batch size=%s" % (str(data['train']['uniform'][0].shape)))
    logger.error(r"    *Biased train batch size=%s" % (str(data['train']['biased'][0].shape)))
    logger.error("###############")


def get_r3_dataset(config: dict, logger: logging.Logger):
    global global_cutoff
    global_cutoff = config['cutoff']

    dataset = load_dataset(
        path=config["path"]
    )
    data = prepare_data(
        dataset=dataset,
        cutoff=config['cutoff'],
    )
    train_test_split(
        data=data,
        train_prop=config['train_prop'],
        validation_prop=config['validation_prop'],
        seed=config["seed"]
    )
    divide_train_data_into_batches(
        data=data,
        n_batches=config['n_batches']
        )
    log_data(
        data=data,
        logger=logger
        )
    return data


def get_cartesian_product(data: torch.Tensor):
    global biased_matrix
    global global_cutoff
    # Calculate Cartesian Product
    user_ids = torch.unique(data[:, 0])
    item_ids = torch.unique(data[:, 1])
    all_pairs = torch.cartesian_prod(user_ids, item_ids)
    users = all_pairs[:, 0].detach().cpu().numpy()
    items = all_pairs[:, 1].detach().cpu().numpy()
    # Is is observed in bias matrix or not
    observed = biased_matrix[users, items]
    observed = (observed > 0) * 1.0
    observed = observed.reshape(-1, 1)
    observed = torch.tensor(observed, dtype=torch.long, device=data.device)
    # Is it unobserved, negative or positive
    imputation_input = biased_matrix[users, items]
    imputation_input = np.where((imputation_input > 0) & (imputation_input <= global_cutoff), -1, imputation_input)
    imputation_input = np.where(imputation_input > global_cutoff, 1, imputation_input)
    imputation_input = imputation_input + 1
    imputation_input = torch.tensor(imputation_input, dtype=torch.int, device=data.device)
    
    return all_pairs, observed, imputation_input 


def get_unobserved_batch(device, batch_size: int):
    """
    Returns a batch of fully uniform data from possible interactions
    The user-item interaction matrix is highly sparse so the returned batch will be unobserved
    """
    user_ids = torch.randint(0, 15400, (batch_size, 1), dtype=torch.long, device=device)
    item_ids = torch.randint(0, 1000, (batch_size, 1), dtype=torch.long, device=device)
    data = torch.concat((user_ids, item_ids), dim=1)
    return data