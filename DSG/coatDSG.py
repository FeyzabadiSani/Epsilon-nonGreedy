import pandas as pd
import logging
import numpy as np
import torch

from sklearn.preprocessing import OneHotEncoder
from DSG.util import train_test_split, divide_train_data_into_batches

item_features_data = None
user_features_data = None
biased_matrix = None
global_cutoff = None
use_features = True

def load_dataset(path: str):
    """
    Loading Raw coat data.
    returns a dictionary contain raw data and features
    """
    global item_features_data
    global user_features_data
    global biased_matrix
    
    dataset = dict()
    dataset['biased_rating_matrix'] = pd.read_csv(path + 'train.ascii', sep=" ", header=None, engine="python").values
    biased_matrix = dataset['biased_rating_matrix']
    dataset['uniform_rating_matrix'] = pd.read_csv(path + 'test.ascii', sep=" ", header=None, engine="python").values
    item_features_data = pd.read_csv(path + 'user_item_features/item_features.ascii', sep=" ", header=None,
                                           engine="python").values
    user_features_data = pd.read_csv(path + 'user_item_features/user_features.ascii', sep=" ", header=None,
                                           engine="python").values
    return dataset

def remove_uniform_from_biased_matrix(dataset: dict):
    """
    In coat dataset there is possibility that uniform data is the same as biased data
    Hence we remove those data from uniform matrix 
    """
    dataset['biased_rating_matrix'][dataset['uniform_rating_matrix'] > 0] = 0


def encode_data(dataset: dict, matrix_name: str, user_encoder: OneHotEncoder,
                item_encoder: OneHotEncoder, cutoff: int):
    """
    Append onehot id of user and item to their correspoinding feature vector
    """
    global user_features_data
    global item_features_data
    global use_features
    rows, cols = np.where(dataset[matrix_name] > 0)
    user_embedding = user_encoder.transform(np.reshape(rows, (-1, 1))).astype(np.uint8).todense()
    item_embedding = item_encoder.transform(np.reshape(cols, (-1, 1))).astype(np.uint8).todense()
    user_features = user_features_data[rows]
    item_features = item_features_data[cols]
    labels = dataset[matrix_name][rows, cols].reshape(-1, 1)
    labels = np.where(labels > cutoff, 1.0, 0.0)
    if use_features:
        data = np.concatenate((user_embedding, item_embedding, user_features, item_features, labels), axis=1)
    else:
        data = np.concatenate((user_embedding, item_embedding, labels), axis=1)
    return data

def encode_unobserved_data(dataset: dict, user_encoder: OneHotEncoder, item_encoder: OneHotEncoder):
    """
    Encode those user-item interactions which are not available during training
    """
    global user_features_data
    global item_features_data
    global use_features
    rating_mat = dataset['uniform_rating_matrix'] + dataset['biased_rating_matrix']
    rows, cols = np.where(rating_mat == 0)
    user_embedding = user_encoder.transform(np.reshape(rows, (-1, 1))).astype(np.uint8).todense()
    item_embedding = item_encoder.transform(np.reshape(cols, (-1, 1))).astype(np.uint8).todense()
    user_features = user_features_data[rows]
    item_features = item_features_data[cols]
    if use_features:
        data = np.concatenate((user_embedding, item_embedding, user_features, item_features), axis=1)
    else:
        data = np.concatenate((user_embedding, item_embedding), axis=1)
    return data


def prepare_data(dataset: dict, cutoff: int):
    """
    Encode raw data to model ready form
    Returns a dictionary contains ready to use data.
    """
    data = dict()
    user_encoder, item_encoder = OneHotEncoder(), OneHotEncoder()
    user_encoder.fit(np.arange(dataset['uniform_rating_matrix'].shape[0]).reshape(-1, 1))
    item_encoder.fit(np.arange(dataset['uniform_rating_matrix'].shape[1]).reshape(-1, 1))
    data['st'] = encode_data(
        dataset=dataset,
        matrix_name="uniform_rating_matrix",
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        cutoff=cutoff,
        )
    data['sc'] = encode_data(
        dataset=dataset,
        matrix_name="biased_rating_matrix",
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        cutoff=cutoff,
        )
    data['sa'] = encode_unobserved_data(
        dataset=dataset,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        )
    return data


def log_data(data:dict, logger:logging.Logger):
    """
    Logging prepared data attributes
    """
    logger.error(r"Dataset Statistics")
    logger.error(r"    *Train uniform data shape=%s" % (str(data['st_train'].shape)))
    logger.error(r"    *Train biased data shape=%s" % (str(data['sc'].shape)))
    logger.error(r"    *Train unobserved data shape=%s" % (str(data['sa'].shape)))
    logger.error(r"    *Validation data shape=%s" % (str(data['st_validation'].shape)))
    logger.error(r"    *Test data shape=%s" % (str(data['st_test'].shape)))
    logger.error(r"    *Number of training batches=%d" % len(data['train']['uniform']))
    logger.error(r"    *Uniform train batch size=%s" % (str(data['train']['uniform'][0].shape)))
    logger.error(r"    *Biased train batch size=%s" % (str(data['train']['biased'][0].shape)))
    logger.error("###############")


def get_coat_dataset(config:dict, logger: logging.Logger):
    """
    Prepare coat dataset for experiments
    """
    global global_cutoff
    global use_features
    global_cutoff = config['cutoff']
    use_features = config['use_features'] if 'use_features' in config else True

    dataset = load_dataset(
        path=config["path"]
    )
    remove_uniform_from_biased_matrix(
        dataset=dataset
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
    global user_features_data
    global item_features_data
    global biased_matrix
    global global_cutoff
    global use_features
    # Extract user_ids
    user_embedding = data[:, :290]
    item_embedding = data[:, 290:590]
    user_ids = torch.where(user_embedding == 1)[1]
    item_ids = torch.where(item_embedding == 1)[1]
    # Calculate cartesian product
    user_ids = torch.unique(user_ids)
    item_ids = torch.unique(item_ids)
    all_pairs = torch.cartesian_prod(user_ids, item_ids)
    # Extract corresponding user and item features
    users = all_pairs[:, 0].detach().cpu().numpy()
    items = all_pairs[:, 1].detach().cpu().numpy()
    if use_features:
        user_features = user_features_data[users]
        item_features = item_features_data[items]
        user_features = torch.tensor(user_features, dtype=data.dtype, device=data.device)
        item_features = torch.tensor(item_features, dtype=data.dtype, device=data.device)
    # Build user and item embeddings of cartesian product
    prod_user_embedding = torch.zeros(
        size=(all_pairs.shape[0], 290),
        dtype=data.dtype,
        device=data.device
    )
    prod_user_embedding[:, users] = 1.0
    prod_item_embedding = torch.zeros(
        size=(all_pairs.shape[0], 300),
        dtype=data.dtype,
        device=data.device
    )
    prod_item_embedding[:, items] = 1.0
    # Is it observed in bias matrix or not
    observed = biased_matrix[users, items]
    observed = (observed > 0) * 1.0
    observed = observed.reshape(-1, 1)
    observed = torch.tensor(observed, dtype=torch.float, device=data.device)
    # Is it unobserved, negative or positive
    imputation_input  = biased_matrix[users, items]
    imputation_input = np.where((imputation_input > 0) & (imputation_input <= global_cutoff), -1, imputation_input)
    imputation_input = np.where(imputation_input > global_cutoff, 1, imputation_input)
    imputation_input = imputation_input + 1
    imputation_input = torch.tensor(imputation_input, dtype=torch.int, device=data.device)
    # Return product data
    if use_features:
        prod_data = torch.cat((prod_user_embedding, prod_item_embedding, user_features, item_features), dim=1)
    else:
        prod_data = torch.cat((prod_user_embedding, prod_item_embedding), dim=1)
    return prod_data, observed, imputation_input
    





