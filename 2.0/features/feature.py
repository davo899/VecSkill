import torch


class Feature:

    def __init__(self, added_length=0, subfeatures=[]):
        self.__added_length = added_length
        self.__subfeatures = subfeatures

    def _added_tensor(self, _):
        return torch.zeros(self.__added_length)

    def length(self):
        return self.__added_length + sum(feature.length() for feature in self.__subfeatures)

    def empty_tensor(self):
        return torch.zeros(self.length())

    def tensor(self, data):
        return torch.cat([self._added_tensor(data)] + [feature.tensor(data) for feature in self.__subfeatures])
