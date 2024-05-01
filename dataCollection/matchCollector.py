import sys
import json

if __name__ == "__main__":
    from riotapi import KEY, request, cursor, conn
    
    with open("next_matchid.txt", "r") as file:
        id_ = int(file.readline())

    server = sys.argv[1]
    prefix = sys.argv[2]

    sys.tracebacklimit = 0
    try:
        while True:
            print()
            if (matchData := request(server, "match/v5/matches/"+prefix+"_"+str(id_), KEY)):
                cursor.execute(
                    "INSERT INTO league.\"Match\" VALUES (" +
                        "'" + json.dumps(matchData) + "'," +
                        str(id_) +
                    ");"
                )
                conn.commit()
            
            id_ -= 1

    finally:
        print("Shutting Down...")

        with open("next_matchid.txt", "w") as file:
            file.write(str(id_))
