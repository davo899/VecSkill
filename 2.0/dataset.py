import torch
from torch import nn
from torch.utils.data import Dataset
import math
from psycopg2 import connect
from constants import DRAFT_PICK, RANKED_SOLO, RANKED_FLEX, BLUE_TEAM
from features import EventFeature
import time
import datetime


EVENT_TYPES = ("CHAMPION_KILL",)


class Match:

    def __init__(self, data):
        info, timeline = data
        self.__info = info["info"]
        self.__timeline = timeline["info"]
        self.__event_count = sum(1 if event["type"] in EVENT_TYPES else 0 for frame in self.__timeline["frames"] for event in frame["events"])

    def __blue_team_won(self):
        for team in self.__info["teams"]:
            if team["win"] and team["teamId"] == BLUE_TEAM:
                return True
        return False

    def event_count(self):
        return self.__event_count

    def timeline_tensor(self):
        event_tensors = []
        previous_event_timestamp = 0
        for frame in self.__timeline["frames"]:
            for event in frame["events"]:
                if event["type"] not in EVENT_TYPES:
                    continue
                event["timeSinceLastEvent"] = event["timestamp"] - previous_event_timestamp
                event_tensors.append(EventFeature().tensor(event))
                previous_event_timestamp = event["timestamp"]

        return torch.stack(event_tensors) if event_tensors else torch.empty((0, EventFeature().length()))

    def result_tensor(self):
        return torch.ones(1) if self.__blue_team_won() else torch.zeros(1)


class TimelineDataset(Dataset):

    def __init__(self, size_limit=math.inf):
        start = time.time()
        self.x = []
        self.y = []
        match_count = 0
        print("Loading dataset")
        with open("dbkey.txt", "r", encoding="utf-8") as file:
            conn_string = file.readline()

        conn = connect(conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(\"ID\") FROM league.\"Match\";")
        minId = cursor.fetchall()[0][0]
        while minId:
            cursor.execute(
                "SELECT \"MatchJson\", \"MatchTimeline\" FROM league.\"Match\" " +
                f"WHERE \"MatchTimeline\" IS NOT NULL AND \"ID\" >= {minId} AND \"ID\" < {minId + 10000} " +
                f"AND \"Queue\" IN ({DRAFT_PICK}, {RANKED_SOLO}, {RANKED_FLEX}) " +
                "AND \"GameMode\" = 'CLASSIC' " +
                "AND \"GameType\" = 'MATCHED_GAME';"
            )
            batch = [match for match in (Match(data) for data in cursor.fetchall()) if match.event_count() > 0]
            match_count += len(batch)
            for match in batch:
                self.x.append(match.timeline_tensor())
                self.y.append(match.result_tensor())

            print(f"{match_count} matches loaded")
            if match_count >= size_limit:
                break

            cursor.execute(f"SELECT MIN(\"ID\") FROM league.\"Match\" WHERE \"ID\" >= {minId + 10000};")
            minId = cursor.fetchall()[0][0]

        event_counts = [tensor.size(0) for tensor in self.x]
        max_event_count = max(event_counts)
        self.x = torch.stack([nn.functional.pad(tensor, (0, 0, 0, max_event_count - tensor.size(0))) for tensor in self.x])
        self.x = (self.x - self.x.mean(dim=(0, 1), keepdim=True)) / (self.x.std(dim=(0, 1), keepdim=True) + 1e-6)
        self.y = torch.stack(self.y).expand(-1, self.x.shape[1])
        self.mask = torch.stack([torch.cat([torch.ones(event_count), torch.zeros(max_event_count - event_count)]) for event_count in event_counts])
        self.length = len(self.x)
        print(f"Loaded {self.length} matches in {datetime.timedelta(seconds=time.time() - start)}")

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]

    def __len__(self):
        return self.length
