from constants import CHAMPION_IDS


class PlayerDTO:

    def __init__(self):
        self.champion = 0
        self.team = 0
        self.kills = 0
        self.deaths = 0
        self.assists = 0

    def to_bytes(self):
        bytes_ = bytearray()

        bytes_.extend(self.champion.to_bytes(1, "little"))
        bytes_.extend(self.team.to_bytes(1, "little"))
        bytes_.extend(self.kills.to_bytes(1, "little"))
        bytes_.extend(self.deaths.to_bytes(1, "little"))
        bytes_.extend(self.assists.to_bytes(1, "little"))

        return bytes_

    def from_bytes(self, bytes_):
        self.champion = int(next(bytes_))
        self.team = int(next(bytes_))
        self.kills = int(next(bytes_))
        self.deaths = int(next(bytes_))
        self.assists = int(next(bytes_))
        return self

    def from_json(self, json_):
        self.champion = CHAMPION_IDS.index(json_["championId"])
        self.team = json_["teamId"]
        self.kills = json_["kills"]
        self.deaths = json_["deaths"]
        self.assists = json_["assists"]
        return self
        

class MatchDTO:

    def __init__(self):
        self.winner = 0
        self.players = [PlayerDTO() for _ in range(10)]

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
