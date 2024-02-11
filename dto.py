from constants import CHAMPION_IDS


class PlayerDTO:

    def to_bytes(self):
        bytes_ = bytearray()

        bytes_.extend(self.champion.to_bytes(1, "little"))
        bytes_.extend(self.team.to_bytes(1, "little"))

        return bytes_

    def from_bytes(self, bytes_):
        self.champion = int(next(bytes_))
        self.team = int(next(bytes_))
        return self

    def from_json(self, json_):
        self.champion = CHAMPION_IDS.index(json_["championId"])
        self.team = json_["teamId"]
        return self
        


class MatchDTO:

    def to_bytes(self):
        bytes_ = bytearray()

        bytes_.extend(self.winner.to_bytes(1, "little"))
        for player in self.players:
            bytes_.extend(player.to_bytes())

        return bytes_

    def from_bytes(self, bytes_):
        self.winner = int(next(bytes_))
        self.players = (PlayerDTO().from_bytes(bytes_) for _ in range(10))
        return self

    def from_json(self, json_):
        for team in json_["teams"]:
            if team["win"]:
                self.winner = team["teamId"]
                break
            
        self.players = (PlayerDTO().from_json(player_json) for player_json in json_["participants"])
        return self
