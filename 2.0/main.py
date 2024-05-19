import torch
from torch import nn
from sklearn.model_selection import train_test_split
from dataset import TimelineDataset
from features import EventFeature
import copy


class FeedForwardModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Linear(EventFeature().length(), 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return torch.sigmoid(torch.cumsum(self.__layers(x).squeeze(), dim=1))

    def accuracy(self, x, y):
        return self(x).round().squeeze().eq(y).sum() / float(y.numel())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = TimelineDataset(size_limit=10_000)
    max_event_count = max(x.size(0) for x, _ in dataset)
    
    X, y, mask = dataset.x.to(device), dataset.y.to(device), dataset.mask.to(device)
    X = (X - X.mean(dim=(0, 1), keepdim=True)) / (X.std(dim=(0, 1), keepdim=True) + 1e-6)
    print("Data scaled")

    X_train_val, X_test, y_train_val, y_test, mask_train_val, mask_test = train_test_split(X, y, mask, test_size=0.2)
    X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(X_train_val, y_train_val, mask_train_val, test_size=0.25)
    print("Dataset split into train, val, test")

    model = FeedForwardModel().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()

    max_acc = -1
    
    NUM_EPOCHS = 3000
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            y_predicted = model(X_train)
            loss = criterion(y_predicted, y_train) * mask_train
            loss = loss.sum() / mask_train.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.no_grad():
            if (acc := model.accuracy(X_val, y_val)) > max_acc:
                max_acc = acc
                max_acc_model = copy.deepcopy(model)

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}, loss = {loss.item():.4f}, val_acc: {acc:.4f}')

    with torch.no_grad():
        print(f'accuracy = {max_acc_model.accuracy(X_test, y_test):.4f}')

if __name__ == "__main__":
    main()
