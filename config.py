import torch

device = torch.device('cuda:0')
# device = torch.device('cpu')

COAT_LOGGER_PATH = "<path-to-coat-logs-directory>"
R3_LOGGER_PATH = "<path-to-YahooR3-logs-directory>"

COAT_DATASET_CONFIG = {
    "path": "<path-to-coat-dataset>",
    "cutoff": 4,
    "n_batches": 1,
    "train_prop": 0.2,
    "validation_prop": 0.4,
    "seed": 888,
    # "use_features": False,
}



R3_DATASET_CONFIG = {
    "path": "<path-to-YahooR3-dataset>",
    "cutoff": 4,
    "n_batches": 1,
    "train_prop": 0.2,
    "validation_prop": 0.4,
    "seed": 888,
}

# You need to activate models that you want to train
# Be cautious as some models need their teachers to be trained first 
TRAIN_ORDER = [
    ('st-net', None, 'classic'),
    ('sc-net', None, 'classic'), 
    ('s-net', None, 'classic'),
    # ('dropout-sc-net', None, 'classic'), 
    # ('dropout-s-net', None, 'classic'), 
    ('l1-net', 'st-net', 'proposed'), 
    ('l2-net', 'st-net', 'proposed'),
    ('kl-net', 'st-net', 'proposed'),
    ('js-net', 'st-net', 'proposed'),
    # ('dropout-l1-net', 'st-net', 'proposed'),
    # ('dropout-l2-net', 'st-net', 'proposed'),
    # ('dropout-kl-net', 'st-net', 'proposed'),
    # ('dropout-js-net', 'st-net', 'proposed'),
    ('refine-net', 'st-net', 'kdrec_refine'),
    ('bridge-sc-net', 'bridge-st-net', 'kdrec_bridge'),
    ('autodebias-net', None, 'autodebias'),
    ('dub-net', 'st-net', 'dub'),
]