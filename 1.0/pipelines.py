import torch
from torch import nn
from dto import MatchDTO
from constants import CHAMPION_IDS, BLUE_TEAM


class Pipeline():

    def __init__(self):
        self._input_features = self.match_to_x(MatchDTO()).shape[0]
    
    def forward(self, input_):
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

    def forward(self, input_):
        player_count = input_.shape[2]
        scores = torch.stack([self.model(input_[:, :, i]) for i in range(player_count)])
        self.scores = scores.clone()
        self.mean = self.scores.mean(dim=0)
        self.var = self.scores.var(dim=0)
        for i in range(player_count // 2):
            scores[(player_count // 2) + i] *= -1
        return torch.sigmoid(scores.sum(dim=0))


class SinglePlayerL2NormPipeline(SinglePlayerPipeline):

    def __init__(self, device, dropout=0):
        super().__init__(device, dropout)
        self.name = "single-player-l2-norm"

    def forward(self, input_):
        player_count = input_.shape[2]
        scores = nn.functional.normalize(
            torch.stack([self.model(input_[:, :, i]) for i in range(player_count)], dim=0),
            p=2,
            dim=0
        )
        self.mean = scores.flatten().mean(dim=0)
        self.var = scores.flatten().var(dim=0)
        for i in range(player_count // 2):
            scores[(player_count // 2) + i] *= -1
        return torch.sigmoid(scores.sum(dim=0))
