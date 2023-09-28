import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):

    def __init__(self, user_embed_dim: int, item_embed_dim: int, hidden_layers_dim: list,
                 dropout_rate: float):
        super().__init__()
        self.user_embedding = nn.Embedding(15400, user_embed_dim)
        self.item_embedding = nn.Embedding(1000, item_embed_dim)
        self.input_dim = user_embed_dim + item_embed_dim

        self.layers = list()
        self.layers.append(nn.Linear(in_features=self.input_dim, out_features=hidden_layers_dim[0]))
        self.layers.append(nn.BatchNorm1d(num_features=hidden_layers_dim[0]))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Dropout(p=dropout_rate))
        for i in range(len(hidden_layers_dim) - 1):
            self.layers.append(
                nn.Linear(in_features=hidden_layers_dim[i], out_features=hidden_layers_dim[i+1])
            )
            self.layers.append(nn.BatchNorm1d(num_features=hidden_layers_dim[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(in_features=hidden_layers_dim[-1], out_features=1))
        self.layers.append(nn.Sigmoid())
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        user_embedding = self.user_embedding(x[:, 0])
        item_embedding = self.item_embedding(x[:, 1])
        output = torch.concat((user_embedding, item_embedding), dim=1)
        for layer in self.layers:
            output = layer(output)
        return output


# Autodebias code is based on https://github.com/DongHande/AutoDebias/blob/main/train_explicit.py
class WeightNetwork(nn.Module):
    """
    Autodebias
    """

    def __init__(self) -> None:
        super().__init__()
        self.user_bias = nn.Embedding(15400, 1)
        self.item_bias = nn.Embedding(1000, 1)
        self.data_bias = nn.Embedding(2, 1)


    def forward(self, x, obs_rew):
        obs_rew = obs_rew.long()
        u_bias = self.user_bias(x[:, 0])
        i_bias = self.item_bias(x[:, 1])
        d_bias = self.data_bias(obs_rew)
        output = u_bias + i_bias + d_bias
        return torch.exp(output / 5)


class ImputationNetwork(nn.Module):
    
    def __init__(self) -> None:
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
                user_embed_dim=models_spec[model_name]['architecture_config']['user_embed_dim'],
                item_embed_dim=models_spec[model_name]['architecture_config']['item_embed_dim'],
                hidden_layers_dim=models_spec[model_name]['architecture_config']['hidden_layers_dim'],
                dropout_rate=models_spec[model_name]['architecture_config']['dropout_rate'],
            )
            models[model_name]['weight1_architecture'] = WeightNetwork()
            models[model_name]['weight2_architecture'] = WeightNetwork()
            models[model_name]["imputation_architecture"] = ImputationNetwork()
        else:
            models[model_name]['architecture'] = FeedForwardNetwork(
                user_embed_dim=models_spec[model_name]['architecture_config']['user_embed_dim'],
                item_embed_dim=models_spec[model_name]['architecture_config']['item_embed_dim'],
                hidden_layers_dim=models_spec[model_name]['architecture_config']['hidden_layers_dim'],
                dropout_rate=models_spec[model_name]['architecture_config']['dropout_rate'], 
            )

        models[model_name]['config'] = models_spec[model_name]['train_config']
    
    return models
