import torch
import ijson
import os
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import copy
import math
import time
from constants import DATASET_FILE, DB_KEY_FILE, DRAFT_PICK, RANKED_SOLO, RANKED_FLEX
from dto import MatchDTO
from psycopg2 import connect
from pipelines import AllPlayersPipeline, SinglePlayerPipeline


class LeagueDataset(Dataset):

    def __stream_dataset_file(self):
        with open(DATASET_FILE, 'rb') as file:
            for match in ijson.items(file, 'item'):
                yield match

    def __add_match(self, match):
        matchDTO = MatchDTO().from_json(match)
        self.x.append(self.__pipeline.match_to_x(matchDTO))
        self.y.append(self.__pipeline.match_to_y(matchDTO))

    def __init__(self, pipeline, use_file=False, size_limit=math.inf):
        self.__pipeline = pipeline
        self.x = []
        self.y = []
        match_count = 0
        print("Loading dataset")

        if use_file:
            for match in self.__stream_dataset_file():
                if len(x_tensors) >= size_limit:
                    break
                self.__add_match(match)
                match_count += 1
                if match_count % 10_000 == 0:
                    print(f"{match_count} matches loaded")
        else:
            with open(DB_KEY_FILE, "r", encoding="utf-8") as file:
                conn_string = file.readline()

            conn = connect(conn_string)
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
            minId = cursor.fetchall()[0][0]
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
                    self.__add_match(match)

                print(f"{match_count} matches loaded")
                if match_count >= size_limit:
                    break

                cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
                minId = cursor.fetchall()[0][0]

        self.x = torch.stack(self.x)
        self.y = torch.stack(self.y)
        self.length = len(self.x)
        print("Dataset loaded")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


def main():
    if torch.cuda.is_available(): 
        dev = "cuda:0"
    else: 
        dev = "cpu"

    device = torch.device(dev)

    pipeline = AllPlayersPipeline(device=device, dropout=0)
    dataset = LeagueDataset(pipeline=pipeline, use_file=False, size_limit=1_000)

    X, y = dataset.x.to(device), dataset.y.to(device)
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
    print("Data scaled")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
    print("Dataset split into train, val, test")

    lr = 0.1
    num_epochs = 2000
    criterion = nn.BCELoss()

    max_acc = -1
    max_acc_pipeline = None
    
    optimizer = torch.optim.SGD(pipeline.model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        y_predicted = pipeline.forward(X_train, device)

        loss = criterion(y_predicted, y_train)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        y_val_predicted = pipeline.forward(X_val, device)
        y_val_predicted_classes = y_val_predicted.round()
        acc = y_val_predicted_classes.eq(y_val).sum() / float(y_val.shape[0])
        if acc > max_acc:
            max_acc = acc
            max_acc_model = copy.deepcopy(pipeline.model)

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}, val_acc: {acc:.4f}')

    with torch.no_grad():
        pipeline.model = max_acc_model
        y_predicted = pipeline.forward(X_test, device)
        y_predicted_classes = y_predicted.round()
        acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
        print(f'accuracy = {acc:.4f}')

        MODELS_DIRECTORY = "savedModels"
        if not os.path.exists(MODELS_DIRECTORY):
            os.makedirs(MODELS_DIRECTORY)
        torch.save(pipeline.model.state_dict(), f"{MODELS_DIRECTORY}/{pipeline.name}-{(acc * 100):.2f}-{len(y)}-{int(time.time())}.pth")

if __name__ == "__main__":
    main()
