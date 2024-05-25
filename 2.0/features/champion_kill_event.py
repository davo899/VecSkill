import torch
from .event_feature import EventFeature


class ChampionKillEventFeature(EventFeature):

    def __init__(self):
        super().__init__(added_length=36)

    def _added_tensor(self, event):
        if event["type"] != "CHAMPION_KILL":
            return self.empty_tensor()

        return torch.cat([
            self._participant_tensor(event["killerId"]),
            self._participant_tensor(event["victimId"]),
            self._assisting_participants_tensor(event["assistingParticipantIds"] if "assistingParticipantIds" in event else []),
            torch.tensor([
                sum(
                    dealt["magicDamage"] + dealt["physicalDamage"] + dealt["trueDamage"]
                    for dealt in (event["victimDamageDealt"] if "victimDamageDealt" in event else [])
                ),
                sum(
                    received["magicDamage"] + received["physicalDamage"] + received["trueDamage"]
                    for received in event["victimDamageReceived"]
                ),
                event["position"]["x"],
                event["position"]["y"],
                event["bounty"],
                event["shutdownBounty"]
            ])
        ])
