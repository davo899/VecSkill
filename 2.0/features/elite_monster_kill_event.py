import torch
from .event_feature import EventFeature


MONSTER_TYPES = ['DRAGON', 'BARON_NASHOR', 'RIFTHERALD']
MONSTER_SUB_TYPES = ['NONE', 'CHEMTECH_DRAGON', 'EARTH_DRAGON', 'ELDER_DRAGON', 'AIR_DRAGON', 'FIRE_DRAGON', 'WATER_DRAGON', 'HEXTECH_DRAGON']

class EliteMonsterKillEventFeature(EventFeature):

    def __init__(self):
        super().__init__(added_length=23 + len(MONSTER_TYPES) + len(MONSTER_SUB_TYPES))

    def _added_tensor(self, event):
        if event["type"] != "ELITE_MONSTER_KILL":
            return self.empty_tensor()

        if "monsterSubType" not in event:
            event["monsterSubType"] = "NONE"

        return torch.cat([
            self._participant_tensor(event["killerId"]),
            self._assisting_participants_tensor(event["assistingParticipantIds"] if "assistingParticipantIds" in event else []),
            self._one_hot(MONSTER_TYPES.index(event["monsterType"]), len(MONSTER_TYPES)),
            self._one_hot(MONSTER_SUB_TYPES.index(event["monsterSubType"]), len(MONSTER_SUB_TYPES)),
            torch.tensor([
                event["position"]["x"],
                event["position"]["y"],
                event["bounty"]
            ])
        ])
