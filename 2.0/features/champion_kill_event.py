import torch
from .feature import Feature


class ChampionKillEventFeature(Feature):

    def __init__(self):
        super().__init__(added_length=36)

    def __assisting_participants_tensor(self, assisting_participants):
        tensor = torch.zeros(10)
        for i in assisting_participants:
            tensor[i - 1] = 1
        return tensor

    def __participant_tensor(self, participant_id):
        tensor = torch.zeros(10)
        tensor[participant_id - 1] = 1
        return tensor

    def _added_tensor(self, event):
        if event["type"] != "CHAMPION_KILL":
            return self.empty_tensor()

        return torch.cat([
            self.__participant_tensor(event["killerId"]),
            self.__participant_tensor(event["victimId"]),
            self.__assisting_participants_tensor(event["assistingParticipantIds"] if "assistingParticipantIds" in event else []),
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
