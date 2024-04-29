import torch
import ijson
import os
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import math
from constants import BLUE_TEAM, CHAMPION_IDS, DATASET_FILE, DB_KEY_FILE, DRAFT_PICK, RANKED_SOLO, RANKED_FLEX, MATCH_COUNT_CUTOFF
from dto import MatchDTO
from psycopg2 import connect


DATASET_SIZE_LIMIT = 400_000
MODELS_DIRECTORY = "models"

def get_player_tensor(playerDTO):
    champion_tensor = torch.zeros(len(CHAMPION_IDS))
    champion_tensor[playerDTO.champion] = 1
    return torch.cat((
        champion_tensor,
        torch.Tensor(playerDTO.count_features)
    ))

def get_match_tensor(matchDTO):
    return torch.cat([get_player_tensor(playerDTO) for playerDTO in sorted(matchDTO.players, key=lambda p: p.team)])

def get_match_result_tensor(matchDTO):
    return torch.ones(1) if matchDTO.winner == BLUE_TEAM else torch.zeros(1)


class LeagueDataset(Dataset):

    def stream_dataset_file(self):
        with open(DATASET_FILE, 'rb') as file:
            for match in ijson.items(file, 'item'):
                yield match

    def __init__(self, use_file):
        x_tensors = []
        y_tensors = []
        print("Loading dataset")

        if use_file:
            for match in self.stream_dataset_file():
                if len(x_tensors) >= DATASET_SIZE_LIMIT:
                    break
                matchDTO = MatchDTO().from_json(match)
                x_tensors.append(get_match_tensor(matchDTO))
                y_tensors.append(get_match_result_tensor(matchDTO))
        else:
            with open(DB_KEY_FILE, "r", encoding="utf-8") as file:
                conn_string = file.readline()

            conn = connect(conn_string)
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
            minId = cursor.fetchall()[0][0]
            match_count = 0
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
                    matchDTO = MatchDTO().from_json(match)
                    x_tensors.append(get_match_tensor(matchDTO))
                    y_tensors.append(get_match_result_tensor(matchDTO))

                print(f"{match_count} matches loaded")
                if match_count >= MATCH_COUNT_CUTOFF:
                    break

                cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
                minId = cursor.fetchall()[0][0]

        self.x = torch.stack(x_tensors)
        self.y = torch.stack(y_tensors)
        self.length = len(self.x)
        print("Dataset loaded")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


INPUT_FEATURES = get_match_tensor(MatchDTO()).shape[0]
MODEL = nn.Sequential(
    nn.Linear(INPUT_FEATURES, 200),
    nn.Tanh(),
    nn.Linear(200, 200),
    nn.Tanh(),
    nn.Linear(200, 50),
    nn.Tanh(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)

def main():
    if torch.cuda.is_available(): 
        dev = "cuda:0"
    else: 
        dev = "cpu"

    device = torch.device(dev)

    dataset = LeagueDataset(use_file=True)

    X, y = dataset.x, dataset.y

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

    model = MODEL.to(device)
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

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}, val_acc: {acc:.4f}')

    with torch.no_grad():
        y_predicted = max_acc_model(X_test)
        y_predicted_classes = y_predicted.round()
        acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
        print(f'accuracy = {acc:.4f}')

        if not os.path.exists(MODELS_DIRECTORY):
            os.makedirs(MODELS_DIRECTORY)
        torch.save(max_acc_model.state_dict(), f"{MODELS_DIRECTORY}/model-{int(acc * 100)}.pth")

if __name__ == "__main__":
    main()
