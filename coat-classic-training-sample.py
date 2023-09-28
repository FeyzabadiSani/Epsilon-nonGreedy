import logging
from config import *
from DSG.coatDSG import *
from runner.coatRunner import run
from util.common import DataType
from util.evaluation import save_results

EXPERIMENT_CONFIG = {
    'name': 'coat-classic-training-schema',
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
        'sc-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 32,
            }
        },
        's-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.001,
                'batch_size': 32,
            }
        },
        'l1-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.01,
            }
        },
        'l2-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 1,
            }
        },
        'kl-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.1,
            }
        },
        'js-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 16,
                'lambda': 0.001,
            }
        },
        'bridge-st-net': {
            'architecture_config': {
                'hidden_layers_dim': [128, 128],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.uniform,
                'weight_decay': 0.01,
                'batch_size': 16,
            },
        },
        'bridge-sc-net': {
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 64,
                'unobserved_batch_size': 32,
            },
        },
        'refine-net': {
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'alpha': 0.2,
            },
        },
        'autodebias-net': {
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.5,
            },
            'train_config': {
                'imputation_weight_decay': 0.001,
                'weights_weight_decay': 0.0001,
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 32,
                'uniform_batch_size': 16,
                'imputation_lambda': 1,
            },
        },
        'dub-net':{
            'architecture_config': {
                'hidden_layers_dim': [256, 256],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 64,
                'unobserved_batch_size': 32,
                'gamma': 0.0001,
            },
        }
    },
    'n_experiments': 5,
    'n_batches': 1,
    'path':  "<path-to-model-saving-directory>",
    'result_path': "<path-to-results-logging-directory>",
    'pick_ratio': 1.0,
    'n_epochs': 200,
    'conv_wait_steps': 5,
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




