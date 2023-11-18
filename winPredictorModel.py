from torch import nn

def winPredictor(input_features):
    return nn.Sequential(
        nn.Linear(input_features, 200),
        nn.Tanh(),
        nn.Linear(200, 200),
        nn.Tanh(),
        nn.Linear(200, 50),
        nn.Tanh(),
        nn.Linear(50, 1),
        nn.Sigmoid()
    )