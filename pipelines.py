import torch
from torch import nn
from dto import MatchDTO
from constants import CHAMPION_IDS, BLUE_TEAM


class Pipeline():

    def __init__(self):
        self._input_features = self.match_to_x(MatchDTO()).shape[0]
    
    def forward(self, input_, device):
        return self.model(input_)

    def match_to_y(self, matchDTO):
        return torch.ones(1) if matchDTO.winner == BLUE_TEAM else torch.zeros(1)

    def _player_tensor(self, playerDTO):
        champion_tensor = torch.zeros(len(CHAMPION_IDS))
        champion_tensor[playerDTO.champion] = 1
        return torch.cat((
            champion_tensor,
            torch.Tensor(playerDTO.count_features)
        ))


class AllPlayersPipeline(Pipeline):
    
    def __init__(self, device, dropout=0):
        super().__init__()
        self.name = "all-players"
        self.model = nn.Sequential(
            nn.Linear(self._input_features, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        ).to(device)
       
    def match_to_x(self, matchDTO):
        return torch.cat([self._player_tensor(playerDTO) for playerDTO in sorted(matchDTO.players, key=lambda p: p.team)])


class SinglePlayerPipeline(Pipeline):

    def __init__(self, device, dropout=0):
        super().__init__()
        self.name = "single-player"
        self.model = nn.Sequential(
            nn.Linear(self._input_features, 200),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(200, 1)
        ).to(device)

    def match_to_x(self, matchDTO):
        return torch.stack([self._player_tensor(playerDTO) for playerDTO in sorted(matchDTO.players, key=lambda p: p.team)], dim=1)

    def forward(self, input_, device):
        y_predicted = torch.zeros((input_.shape[0], 1)).to(device)
        for i in range(input_.shape[2]):
            output = self.model(input_[:, :, i])
            if i < 5:
                y_predicted += output
            else:
                y_predicted -= output

        return torch.sigmoid(y_predicted)
