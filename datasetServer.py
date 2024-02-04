import socket
from psycopg2 import connect
from dto import MatchDTO

PORT = 24629
MATCH_COUNT_CUTOFF = 10000
DRAFT_PICK = 400
RANKED_SOLO = 420
RANKED_FLEX = 440

matches = b''
match_count = 0
with open("dbkey.txt", "r") as file:
    conn_string = file.readline()

conn = connect(conn_string)
cursor = conn.cursor()
cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
minId = cursor.fetchall()[0][0]
while minId:
    cursor.execute(f"SELECT \"MatchJson\" FROM league.\"Match\" WHERE \"ID\" >= {minId} AND \"ID\" < {minId + 10000};")
    batch = [
        game for game in (game[0]["info"] for game in cursor.fetchall())
        if game["queueId"] in (DRAFT_PICK, RANKED_SOLO, RANKED_FLEX) and
           game["gameMode"] in ("CLASSIC",) and
           game["gameType"] in ("MATCHED_GAME",)
    ]
    match_count += len(batch)
    matches = matches.join(MatchDTO().from_json(match).to_bytes() for match in batch)
    cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
    minId = cursor.fetchall()[0][0]
    if match_count >= MATCH_COUNT_CUTOFF:
        break
    
print(match_count, "matches loaded")

with socket.socket() as s:
    s.bind(('', PORT))
    s.listen()

    while True:
        client_socket, address = s.accept()
        print("Opened connection with", address)
        client_socket.send(matches)
        client_socket.close()
        print("Closed connection with", address)
