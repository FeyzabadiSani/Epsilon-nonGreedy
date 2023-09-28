import numpy as np


def train_test_split(data:dict, train_prop: float, validation_prop: float,
                     seed: int):
    """
    Divide dataset into 3parts (Train, Validation, Test)
    """
    n = data['st'].shape[0]
    train_offset = int(n * train_prop)
    val_offset = train_offset + int(n * validation_prop)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, n, replace=False)
    data['st_train'] = data['st'][idx[:train_offset], :]
    data['st_validation'] = data['st'][idx[train_offset:val_offset], :]
    data['st_test'] = data['st'][idx[val_offset:], :]


def divide_train_data_into_batches(data: dict, n_batches: int):
    """
    Divide training data into N batches
    """
    train_data = {'uniform': [], 'biased': []}
    uniform_batch_size = data['st_train'].shape[0] // n_batches
    biased_batch_size = data['sc'].shape[0] // n_batches
    i = -1
    for i in range(n_batches - 1):
        train_data['uniform'].append(data['st_train'][i * uniform_batch_size:(i + 1) * uniform_batch_size, :])
        train_data['biased'].append(data['sc'][i * biased_batch_size:(i + 1) * biased_batch_size, :])
    train_data['uniform'].append(data['st_train'][(i + 1) * uniform_batch_size:, :])
    train_data['biased'].append(data['sc'][(i + 1) * biased_batch_size:, :])
    data['train'] = train_data