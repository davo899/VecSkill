import socket
from psycopg2 import connect
from dto import MatchDTO
import math
from constants import DRAFT_PICK, RANKED_SOLO, RANKED_FLEX

PORT = 24629
NEXT_MESSAGE = b"NEXT"
END_MESSAGE = b"END"
MATCH_COUNT_CUTOFF = 1

if __name__ == "__main__":
    matches = []
    match_count = 0
    with open("dbkey.txt", "r") as file:
        conn_string = file.readline()

    conn = connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
    minId = cursor.fetchall()[0][0]
    print("Loading dataset")
    while minId:
        cursor.execute(f"SELECT \"MatchJson\" FROM league.\"Match\" WHERE \"ID\" >= {minId} AND \"ID\" < {minId + 10000};")
        matches += [
            MatchDTO().from_json(match).to_bytes() for match in (match[0]["info"] for match in cursor.fetchall())
            if match["queueId"] in (DRAFT_PICK, RANKED_SOLO, RANKED_FLEX) and
            match["gameMode"] in ("CLASSIC",) and
            match["gameType"] in ("MATCHED_GAME",)
        ]
        cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
        minId = cursor.fetchall()[0][0]
        match_count = len(matches)
        print(f"{match_count} matches loaded")
        if match_count >= MATCH_COUNT_CUTOFF:
            break

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.bind(('', PORT))
            s.listen()
            print(f"Listening on port {PORT}")

            while True:
                client_socket, address = s.accept()
                print(f"Opened connection with {address}")
                try:
                    for match in matches:
                        if client_socket.recv(len(NEXT_MESSAGE)) == NEXT_MESSAGE:
                            client_socket.send(match)
                    if client_socket.recv(len(NEXT_MESSAGE)) == NEXT_MESSAGE:
                        client_socket.send(END_MESSAGE)
                except ConnectionResetError:
                    print("Connection reset by peer")

                client_socket.close()
                print(f"Closed connection with {address}")

        except KeyboardInterrupt:
            print("Shutting down")
