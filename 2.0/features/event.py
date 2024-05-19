import torch
from .feature import Feature
from .champion_kill_event import ChampionKillEventFeature


class EventFeature(Feature):

    def __init__(self):
        super().__init__(added_length=1, subfeatures=[
            ChampionKillEventFeature()
        ])

    def _added_tensor(self, event):
        return torch.tensor([event["timeSinceLastEvent"]])
