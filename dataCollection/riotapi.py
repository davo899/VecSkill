import psycopg2
import json
import time
import requests

class APIKey:

    def __init__(self, keys):
        self.keys = keys
        self.key = -1
        self.gap = (120/len(self.keys)) / 100
        self.time = time.time()

    def getKey(self):
        sleepTime = self.gap - (time.time() - self.time)
        if sleepTime > 0:
            time.sleep(sleepTime)

        self.time = time.time()
        return self.keys[self.key]

    def prevKey(self):
        self.key -= 1
        self.key %= len(self.keys)

    def changeKey(self):
        self.key += 1
        self.key %= len(self.keys)
        

def connectDB():
    with open("dbkey.txt", "r", encoding="utf-8") as file:
        conn_string = file.read()
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    return cursor, conn

def request(server: str, url: str, key: APIKey, params: dict = {}):
    key.changeKey()
    tries = 0
    complete = False
    
    while not complete:
        try:
            start = time.time()
            while (r := requests.get("https://"+server+".api.riotgames.com/lol/"+url+"?api_key="+KEY.getKey(), params).text):
                try:
                    r = json.loads(r)
                except json.decoder.JSONDecodeError:
                    print("Error decoding json:")
                    print(r)
                    return None
                    
                timeTaken = str(round(time.time() - start, 2)) + "s"
                
                if isinstance(r, dict) and "status" in r.keys():
                    print(server, url, r["status"]["status_code"], timeTaken)
                    if r["status"]["status_code"] == 503:
                        time.sleep(5)
                        continue
                    
                    tries += 1
                    if tries > 3:
                        return None
                    
                    if r["status"]["status_code"] in (400, 404, 429):
                        return None

                    if r["status"]["status_code"] == 403:
                        raise Exception
                    
                else:
                    print(server, url, 200, timeTaken)
                    complete = True
                    break
                
        except requests.exceptions.ConnectionError:
            print("Could not connect")
            
    return r


cursor, conn = connectDB()

with open("apikey.txt", "r") as file:
    KEY = APIKey(file.readlines())
