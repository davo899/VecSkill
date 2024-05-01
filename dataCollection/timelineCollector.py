import sys
import json

if __name__ == "__main__":
    from riotapi import KEY, request, cursor, conn

    sys.tracebacklimit = 0
    try:
        while True:
            cursor.execute("SELECT \"ID\" FROM league.\"Match\" WHERE \"MatchTimeline\" IS NULL LIMIT 10000;")
            ids = cursor.fetchall()[0]
            if len(ids) == 0:
                break
            for id_ in ids:
                print()
                if (timeline := request("europe", f"match/v5/matches/EUW1_{id_}/timeline", KEY)):
                    cursor.execute(
                         "UPDATE league.\"Match\" " +
                        f"SET \"MatchTimeline\" = '{json.dumps(timeline)}' " +
                        f"WHERE \"ID\" = {id_};"
                    )
                    conn.commit()

    finally:
        print("Shutting Down...")
