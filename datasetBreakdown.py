import psycopg2


class Categoriser():

    def __init__(self, extractor):
        self.dict = {}
        self.extractor = extractor

    def add(self, item):
        key = self.extractor(item)
        self.dict[key] = 1 + (self.dict[key] if key in self.dict else 0)

    def get(self, key):
        return self.dict[key]


categorisers = {
    "queue": Categoriser(lambda game: game["queueId"]),
    "gameMode": Categoriser(lambda game: game["gameMode"]),
    "gameType": Categoriser(lambda game: game["gameType"]),
    "gameVersion": Categoriser(lambda game: game["gameVersion"]),
    "champion": Categoriser(lambda participant: participant["championName"])
}

with open("dbkey.txt", "r") as file:
    conn_string = file.readline()
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
    
    minId = cursor.fetchall()[0][0]
    gameCount = 0
    while minId:
        cursor.execute(f"SELECT \"MatchJson\" FROM league.\"Match\" WHERE \"ID\" >= {minId} AND \"ID\" < {minId + 10000};")
        games = [game[0]["info"] for game in cursor.fetchall()]
        games = [game for game in games if game["gameType"] == "MATCHED_GAME"] # Matched game type filter
        games = [game for game in games if game["gameMode"] == "CLASSIC"] # Classic game mode filter
        games = [game for game in games if game["queueId"] in (400, 420, 440)] # Draft Pick, Ranked Solo or Ranked Flex queue filter
        
        gameCount += len(games)
        for game in games:
            for participant in game["participants"]:
                categorisers["champion"].add(participant)

        cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
        minId = cursor.fetchall()[0][0]
        print(f"{gameCount} samples loaded")

print(categorisers["champion"].dict)
