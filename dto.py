from constants import CHAMPION_IDS


class PlayerDTO:

    def to_bytes(self):
        return bytes([self.champion])

    def from_bytes(self, bytes_):
        self.champion = next(bytes_)
        self.team = next(bytes_)
        return self

    def from_json(self, json_):
        self.champion = CHAMPION_IDS.index(json_["championId"])
        self.team = json_["teamId"]
        return self
        


class MatchDTO:

    def to_bytes(self):
        return b''.join(player.to_bytes() for player in self.players)

    def from_bytes(self, bytes_):
        self.winner = next(bytes_)
        self.players = (PlayerDTO().from_bytes(bytes_) for _ in range(10))
        return self

    def from_json(self, json_):
        for team in json_["teams"]:
            if team["win"]:
                self.winner = team["teamId"]
                break
            
        self.players = (PlayerDTO().from_json(player_json) for player_json in json_["participants"])
        return self
