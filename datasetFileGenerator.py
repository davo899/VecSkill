import json
from psycopg2 import connect
from constants import DRAFT_PICK, RANKED_SOLO, RANKED_FLEX, DATASET_FILE, DB_KEY_FILE, MATCH_COUNT_CUTOFF

PORT = 24629
NEXT_MESSAGE = b"NEXT"
END_MESSAGE = b"END"

if __name__ == "__main__":
    with open(DB_KEY_FILE, "r", encoding="utf-8") as file:
        conn_string = file.readline()

    conn = connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
    minId = cursor.fetchall()[0][0]

    with open(DATASET_FILE, "w", encoding="utf-8") as file:
        print("Generating dataset file")
        file.write("[")

        match_count = 0
        first = True
        while minId:

            cursor.execute(f"SELECT \"MatchJson\" FROM league.\"Match\" WHERE \"ID\" >= {minId} AND \"ID\" < {minId + 10000};")
            batch = [
                match for match in (match[0]["info"] for match in cursor.fetchall())
                if match["queueId"] in (DRAFT_PICK, RANKED_SOLO, RANKED_FLEX) and
                match["gameMode"] in ("CLASSIC",) and
                match["gameType"] in ("MATCHED_GAME",)
            ]
            match_count += len(batch)
            for match in batch:
                if first:
                    first = False
                else:
                    file.write(", ")
                file.write(json.dumps(match))

            print(f"{match_count} matches written")
            if match_count >= MATCH_COUNT_CUTOFF:
                break

            cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
            minId = cursor.fetchall()[0][0]

        file.write("]")
