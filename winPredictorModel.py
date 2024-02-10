from torch import nn
from constants import CHAMPION_IDS

MODEL = nn.Sequential(
    nn.Linear(2 * len(CHAMPION_IDS), 200),
    nn.Tanh(),
    nn.Linear(200, 200),
    nn.Tanh(),
    nn.Linear(200, 50),
    nn.Tanh(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)
