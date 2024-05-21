import torch
from torch import nn


class FeedForwardModel(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Linear(input_shape[2], 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.__event_index_weights = torch.tensor([(i + 1) ** -1 for i in range(input_shape[1])]).view(1, -1)
        self.__criterion = nn.BCEWithLogitsLoss(reduction='none')

    def to(self, device):
        self.__event_index_weights = self.__event_index_weights.to(device)
        return super().to(device)

    def forward(self, x):
        return torch.sigmoid(torch.cumsum(self.__layers(x).squeeze(), dim=1))

    def accuracy(self, X, y, mask):
        with torch.no_grad():
            return (self(X).round().squeeze().eq(y) * mask).sum() / mask.sum()

    def run_training_step(self, X, y, mask):
        with torch.cuda.amp.autocast():
            return (self.__criterion(self(X), y) * self.__event_index_weights * mask).sum() / mask.sum()
