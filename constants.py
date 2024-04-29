import math

CHAMPION_IDS = (
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
    23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
    42,43,44,45,48,50,51,53,54,55,56,57,58,59,60,61,62,63,64,
    67,68,69,72,74,75,76,77,78,79,80,81,82,83,84,85,86,89,90,
    91,92,96,98,99,101,102,103,104,105,106,107,110,111,112,113,
    114,115,117,119,120,121,122,126,127,131,133,134,136,141,142,
    143,145,147,150,154,157,161,163,164,166,200,201,202,203,221,
    222,223,233,234,235,236,238,240,245,246,254,266,267,268,350,
    360,412,420,421,427,429,432,497,498,516,517,518,523,526,555,
    711,777,875,876,887,888,895,897,902,950
)

DRAFT_PICK = 400
BLIND_PICK = 440
RANKED_SOLO = 420
RANKED_FLEX = 440

BLUE_TEAM = 100
RED_TEAM = 200

DB_KEY_FILE = "dbkey.txt"
DATASET_FILE = "match_dataset.json"
MATCH_COUNT_CUTOFF = math.inf
