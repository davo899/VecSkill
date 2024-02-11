from torch import nn
from constants import CHAMPION_IDS

INPUT_FEATURES = 10 * len(CHAMPION_IDS)

MODEL = nn.Sequential(
    nn.Linear(INPUT_FEATURES, 200),
    nn.Tanh(),
    nn.Linear(200, 200),
    nn.Tanh(),
    nn.Linear(200, 50),
    nn.Tanh(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)
