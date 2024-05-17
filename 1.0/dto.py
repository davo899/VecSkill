from constants import CHAMPION_IDS


class PlayerDTO:

    def __init__(self):
        self.champion = 0
        self.team = 0
        self.count_feature_count = 30
        self.count_features = [0 for _ in range(self.count_feature_count)]

    def to_bytes(self):
        bytes_ = bytearray()

        bytes_.extend(self.champion.to_bytes(1, "little"))
        bytes_.extend(self.team.to_bytes(1, "little"))
        for count_feature in self.count_features:
            bytes_.extend(count_feature.to_bytes(4, "little"))

        return bytes_

    def from_bytes(self, bytes_):
        self.champion = next(bytes_)
        self.team = next(bytes_)
        self.count_features = [
            int.from_bytes(
                bytearray([next(bytes_), next(bytes_), next(bytes_), next(bytes_)]),
                "little"
            )
            for _ in range(self.count_feature_count)
        ]
        return self

    def from_json(self, json_):
        self.champion = CHAMPION_IDS.index(json_["championId"])
        self.team = json_["teamId"]

        challenges = json_["challenges"]
        self.count_features = [
            #json_["kills"],
            #json_["deaths"],
            json_["assists"],

            #json_["goldEarned"],
            #json_["champExperience"],

            json_["totalDamageDealtToChampions"],
            json_["totalTimeCCDealt"],

            json_["allInPings"],
            json_["assistMePings"],
            json_["baitPings"],
            json_["basicPings"],
            json_["commandPings"],
            json_["dangerPings"],
            json_["enemyMissingPings"],
            json_["enemyVisionPings"],
            json_["getBackPings"],
            json_["holdPings"],
            json_["needVisionPings"],
            json_["onMyWayPings"],
            json_["pushPings"],
            json_["visionClearedPings"],
            
            json_["totalMinionsKilled"],
            json_["timePlayed"],
            json_["visionScore"],
            #json_["turretTakedowns"],

            challenges["skillshotsDodged"],
            challenges["skillshotsHit"],

            challenges["dragonTakedowns"],
            challenges["baronTakedowns"],
            challenges["riftHeraldTakedowns"],

            challenges["epicMonsterSteals"],
            challenges["controlWardsPlaced"],
            challenges["bountyGold"],
            challenges["turretPlatesTaken"],
            challenges["unseenRecalls"],
        ]
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
