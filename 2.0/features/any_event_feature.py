import torch
from .feature import Feature
from .champion_kill_event import ChampionKillEventFeature
from .elite_monster_kill_event import EliteMonsterKillEventFeature


class AnyEventFeature(Feature):

    def __init__(self):
        super().__init__(added_length=1, subfeatures=[
            ChampionKillEventFeature(),
            EliteMonsterKillEventFeature()
        ])

    def _added_tensor(self, event):
        return torch.tensor([event["timeSinceLastEvent"]])
