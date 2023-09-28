import logging
from config import *
from DSG.r3DSG import *
from runner.r3Runner import run
from util.common import DataType
from util.evaluation import save_results

EXPERIMENT_CONFIG = {
    'name': 'YahooR3-classic-training-schema',
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
        'sc-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 32,
            },
        },
        's-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.001,
                'batch_size': 32,
            },
        },
        'l1-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 1,
            },
        },
        'l2-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 1,
            },
        },
        'kl-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.7,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.1,
            },
        },
        'js-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.7,
            },
            'train_config': {
                'data_type': DataType.both,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'unobserved_batch_size': 64,
                'lambda': 0.1,
            },
        },
        'bridge-st-net':  {
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
        'bridge-sc-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.8,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 32,
                'unobserved_batch_size': 32,
            },
        },
        'refine-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.7,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'alpha': 0.4,
            },
        },
        'autodebias-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.7,
            },
            'train_config': {
                'imputation_weight_decay': 0.01,
                'weights_weight_decay': 0.01,
                'data_type': DataType.biased,
                'weight_decay': 0.0001,
                'batch_size': 32,
                'uniform_batch_size': 32,
                'imputation_lambda': 0.01,
            },
        },
        'dub-net': {
            'architecture_config': {
                'user_embed_dim': 200,
                'item_embed_dim': 200,
                'hidden_layers_dim': [128, 128, 128],
                'dropout_rate': 0.9,
            },
            'train_config': {
                'data_type': DataType.biased,
                'weight_decay': 0.001,
                'batch_size': 32,
                'unobserved_batch_size': 16,
                'gamma': 0.01,
            },
        }
    },
    'n_experiments': 5,
    'n_batches': 1, # When using classic training schema there is only one batch in training data
    'path': "<path-to-model-saving-directory>",
    'result_path': "<path-to-results-logging-directory>",
    'pick_ratio': 1.0, # When using classic training schema we use all data
    'n_epochs': 200,
    'conv_wait_steps': 20,
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




