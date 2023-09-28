import logging
from DSG.coatDSG import *
from config import COAT_LOGGER_PATH
from runner.coatRunner import run
from util.common import DataType
from util.evaluation import save_results

n_batches = 10
uniform_ratio = 0.05 
pickup_ratio = 0.3

COAT_DATASET_CONFIG = {
    "path": "<path-to-coat-dataset>",
    "cutoff": 4,
    "n_batches": n_batches,
    "train_prop": uniform_ratio,
    "validation_prop": (1 - uniform_ratio) / 2,
    "seed": 888
}

EXPERIMENT_CONFIG = {
    'name': f'coat-sequential-training-sample',
    'models_spec': {
        'st-net':{
            'architecture_config': {
                'hidden_layers_dim': [128, 128],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.uniform,
                'weight_decay': 0.01,
                'batch_size': 16,
            }
        },
        'dropout-sc-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 32,
                'n_samples': 10,
            }
        },
        'dropout-s-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.001,
                'batch_size': 32,
                'n_samples': 10,
            }
        },
        'dropout-l1-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.01,
                'n_samples': 10,
            }
        },
        'dropout-l2-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 1,
                'n_samples': 10,
            }
        },
        'dropout-kl-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.1,
                'n_samples': 10,
            }
        },
        'dropout-js-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.1,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 16,
                'lambda': 0.001,
                'n_samples': 10,
            }
        },
    },
    'n_experiments': 5,
    'n_batches': n_batches,
    'path':  "<path-to-model-saving-directory>",
    'result_path': "<path-to-results-logging-directory>",
    'pick_ratio': pickup_ratio,
    'n_epochs': 200,
    'conv_wait_steps': 10,
}

# Config Where to save Logs
logger = logging.getLogger(EXPERIMENT_CONFIG['name'])
handler = logging.FileHandler(COAT_LOGGER_PATH + EXPERIMENT_CONFIG['name'] + '.log', mode='w')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# Loading data
dataset = get_coat_dataset(
    config=COAT_DATASET_CONFIG,
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
    data_config=COAT_DATASET_CONFIG,
)              
print(auc, logloss)




