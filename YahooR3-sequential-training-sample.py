import logging
from DSG.r3DSG import *
from config import R3_LOGGER_PATH
from runner.r3Runner import run
from util.common import DataType
from util.evaluation import save_results

n_batches = 10 # number of training batches in sequential training schema
uniform_ratio = 0.2 # How much of uniformly collected data should be used for training
pickup_ratio = 0.3  # How much on biased data should be selected by the student in each round of training

R3_DATASET_CONFIG = {
    "path": "<path-to-YahooR3-dataset>",
    "cutoff": 4,
    "n_batches": n_batches,
    "train_prop": uniform_ratio,
    "validation_prop": (1 - uniform_ratio) / 2,
    "seed": 888
}

EXPERIMENT_CONFIG = {
    'name': f'YahooR3-sequential-training-schema',
    'models_spec': {
        'st-net': {
            'architecture_config': {
                'user_embed_dim': 20,
                'item_embed_dim': 50,
                'hidden_layers_dim': [64, 64],
                'dropout_rate': 0.7,
            },
            'train_config': {
                'data_type': DataType.uniform,
                'weight_decay': 0.01,
                'batch_size': 32,
            },
        },
        'dropout-l1-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 1,
                'n_samples': 10,
            },
        },
        'dropout-l2-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 1,
                'n_samples': 10,
            },
        },
        'dropout-kl-net':{
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.1,
                'n_samples': 10,
            },
        },
        'dropout-js-net':{
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.1,
                'n_samples': 10,
            },
        },
    },
    'n_experiments': 1,
    'n_batches': n_batches,
    'path': "<path-to-model-saving-directory>",
    'result_path': "<path-to-results-logging-directory>",
    'pick_ratio': pickup_ratio,
    'n_epochs': 200,
    'conv_wait_steps': 10,
}

# Config Where to save Logs
logger = logging.getLogger(EXPERIMENT_CONFIG['name'])
handler = logging.FileHandler(R3_LOGGER_PATH + EXPERIMENT_CONFIG['name'] + '.log', mode='w')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# Loading data
dataset = get_r3_dataset(
    config=R3_DATASET_CONFIG,
    logger=logger,
)
# running the experiments
auc, logloss = run(
                logger=logger,
                config=EXPERIMENT_CONFIG,
                data=dataset
                )
# Saving results     
save_results(
    auc=auc,
    logloss=logloss,
    experiment_config=EXPERIMENT_CONFIG,
    data_config=R3_DATASET_CONFIG,
)              
print(auc, logloss)




