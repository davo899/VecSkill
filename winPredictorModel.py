from torch import nn

def winPredictor(input_features):
    return nn.Sequential(
        nn.Linear(input_features, 200),
        nn.Tanh(),
        nn.Linear(200, 200),
        nn.Tanh(),
        nn.Linear(200, 1),
        nn.Sigmoid()
    )