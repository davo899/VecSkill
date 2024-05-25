import torch
from .feature import Feature


class EventFeature(Feature):

    def _assisting_participants_tensor(self, assisting_participants):
        tensor = torch.zeros(10)
        for i in assisting_participants:
            tensor[i - 1] = 1
        return tensor

    def _participant_tensor(self, participant_id):
        return self._one_hot(participant_id - 1, 10)
