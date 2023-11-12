import torch
from torch import nn
from winPredictorModel import winPredictor
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import psycopg2
import copy


DRAFT_PICK = 400
BLIND_PICK = 440
RANKED_SOLO = 420
RANKED_FLEX = 440

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

BLUE_TEAM = 100
RED_TEAM = 200


if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu"

device = torch.device(dev)


def get_champions_tensor(game):
    blue_champions_tensor = torch.zeros(len(CHAMPION_IDS))
    red_champions_tensor = torch.zeros(len(CHAMPION_IDS))
    for participant in game["participants"]:
        champion_id = participant["championId"]
        if champion_id in CHAMPION_IDS:
            (blue_champions_tensor if participant["teamId"] == BLUE_TEAM else red_champions_tensor)[CHAMPION_IDS.index(champion_id)] = 1

    return torch.cat((blue_champions_tensor, red_champions_tensor))

def get_result_tensor(game):
    for team in game["teams"]:
        if team["win"]:
            return torch.ones(1) if team["teamId"] == BLUE_TEAM else torch.zeros(1)

    return torch.zeros(1)


class LeagueDataset(Dataset):

    def __init__(self, device, queues, gamemodes, gameTypes):
        with open("dbkey.txt", "r") as file:
            conn_string = file.readline()

        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        self.x = torch.empty(0)
        self.y = torch.empty(0)
        cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
        minId = cursor.fetchall()[0][0]
        while minId:
            cursor.execute(f"SELECT \"MatchJson\" FROM league.\"Match\" WHERE \"ID\" >= {minId} AND \"ID\" < {minId + 10000};")
            batch = list(map(lambda game: game[0]["info"], cursor.fetchall()))
            batch = filter(lambda game: game["queueId"] in queues, batch)
            batch = filter(lambda game: game["gameMode"] in gamemodes, batch)
            batch = filter(lambda game: game["gameType"] in gameTypes, batch)
            batch = list(batch)
            self.x = torch.cat((self.x, torch.stack([get_champions_tensor(game) for game in batch])))
            self.y = torch.cat((self.y, torch.stack([get_result_tensor(game) for game in batch])))

            cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
            minId = cursor.fetchall()[0][0]
            print(f"{len(self.x)} samples loaded")

        self.length = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


dataset = LeagueDataset(
    device,
    (DRAFT_PICK, RANKED_SOLO),
    ("CLASSIC",),
    ("MATCHED_GAME",)
)

X, y = dataset.x, dataset.y

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
X_val = torch.from_numpy(X_val.astype(np.float32)).to(device)
X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)

y_train = y_train.view(y_train.shape[0], 1).to(device)
y_val = y_val.view(y_val.shape[0], 1).to(device)
y_test = y_test.view(y_test.shape[0], 1).to(device)

lr = 0.1
num_epochs = 3000
criterion = nn.BCELoss()

max_acc = -1
max_acc_model = None

model = winPredictor(n_features)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=3e-4, total_iters=num_epochs)

for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    y_val_predicted = model(X_val)
    y_val_predicted_classes = y_val_predicted.round()
    acc = y_val_predicted_classes.eq(y_val).sum() / float(y_val.shape[0])
    if acc > max_acc:
        max_acc = acc
        max_acc_model = copy.deepcopy(model)

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}, val_acc: {acc:.4f}')

with torch.no_grad():
    y_predicted = max_acc_model(X_test)
    y_predicted_classes = y_predicted.round()
    acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
    
    torch.save(max_acc_model.state_dict(), f"model-{int(acc * 100)}.pth")
