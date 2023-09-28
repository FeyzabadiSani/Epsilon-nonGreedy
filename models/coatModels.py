import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):

    def __init__(self, data_dim: int, hidden_layers_dim: list, dropout_rate: float):
        super().__init__()
        self.input_dim = data_dim
        self.n_hidden_layers = len(hidden_layers_dim)
        self.layers = list()
        self.hidden_layers_dim = hidden_layers_dim
        self.dropout_rate = dropout_rate

        self.layers.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_layers_dim[0]))
        self.layers.append(nn.BatchNorm1d(num_features=self.hidden_layers_dim[0]))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Dropout(p=self.dropout_rate))
        for i in range(self.n_hidden_layers - 1):
            self.layers.append(
                nn.Linear(in_features=self.hidden_layers_dim[i], out_features=self.hidden_layers_dim[i + 1]))
            self.layers.append(nn.BatchNorm1d(num_features=self.hidden_layers_dim[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=self.dropout_rate))
        self.layers.append(nn.Linear(in_features=self.hidden_layers_dim[-1], out_features=1))
        self.layers.append(nn.Sigmoid())
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        output = x
        for layer in self.model:
            output = layer(output)
        return output


# Autodebias code is based on https://github.com/DongHande/AutoDebias/blob/main/train_explicit.py

class WeightNetwork(nn.Module):
    """
    AutoDebias
    """
    def __init__(self, data_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_features=data_dim + 1, out_features=1) # 1 is for r/o

    def forward(self, x, obs_rew):
        inp = torch.cat((x, obs_rew), dim=1)
        output = self.layer(inp)
        return torch.exp(output / 5)


class ImputationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_bias = nn.Embedding(3, 1)

    def forward(self, x):
        output = self.data_bias(x)
        return torch.tanh(output)


def initialize_models(models_spec: dict):
    models = dict()
    for model_name in models_spec:
        models[model_name] = {}

        # Autodebias Net needs weights and imputation networks.
        if model_name == 'autodebias-net':
            models[model_name]['architecture'] = FeedForwardNetwork(
                data_dim=637,
                # data_dim=590,
                hidden_layers_dim=models_spec[model_name]['architecture_config']['hidden_layers_dim'],
                dropout_rate=models_spec[model_name]['architecture_config']['dropout_rate']
            )
            models[model_name]['weight1_architecture'] = WeightNetwork(
                data_dim=637,
                # data_dim=590,
            )
            models[model_name]['weight2_architecture'] = WeightNetwork(
                data_dim=637,
                # data_dim=590,
            )
            models[model_name]['imputation_architecture'] = ImputationNetwork()
        else:
            models[model_name]['architecture'] = FeedForwardNetwork(
                data_dim=637,
                # data_dim=590,
                hidden_layers_dim=models_spec[model_name]['architecture_config']['hidden_layers_dim'],
                dropout_rate=models_spec[model_name]['architecture_config']['dropout_rate']
            )
            
        models[model_name]['config'] = models_spec[model_name]['train_config']
    
    return models
