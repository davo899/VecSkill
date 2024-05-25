import torch
from torch import nn
from sklearn.model_selection import train_test_split
from dataset import TimelineDataset
from torch.utils.data import DataLoader, random_split
from models import FeedForwardModel, TransformerModel
import copy
import array


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ratio = 20 / 30
    val_ratio = 9 / 30
    assert train_ratio + val_ratio <= 1

    dataset = TimelineDataset(size_limit=30_000)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print("Dataset split into train, val, test")
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data loaders initialized")

    model = TransformerModel(
        input_dim=dataset.x.shape[2],
        embed_dim=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=128,
        max_event_count=dataset.x.shape[1]
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
    scaler = torch.cuda.amp.GradScaler()

    max_acc = -1
    
    NUM_EPOCHS = 100
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X, y, mask in train_loader:
            optimizer.zero_grad()

            loss = model.run_training_step(X.to(device), y.to(device), mask.to(device))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        loss = epoch_loss / len(train_loader)

        model.eval()
        if (acc := sum(model.accuracy(X.to(device), y.to(device), mask.to(device)) for X, y, mask in val_loader) / len(val_loader)) > max_acc:
            max_acc = acc
            max_acc_model = copy.deepcopy(model)

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch + 1}, loss = {loss:.4f}, val_acc: {acc:.4f}')

    print(f'accuracy = {sum(max_acc_model.accuracy(X.to(device), y.to(device), mask.to(device)) for X, y, mask in test_loader) / len(test_loader):.4f}')
    
    X_test = []
    Y_test = []
    mask_test = []
    for X, Y, masks in test_loader:
        for x in X:
            X_test.append(x)
        for y in Y:
            Y_test.append(y)
        for mask in masks:
            mask_test.append(mask)
    X_test = torch.stack(X_test).to(device)
    Y_test = torch.stack(Y_test).to(device)
    mask_test = torch.stack(mask_test).to(device)

    print(X_test.shape, Y_test.shape, mask_test.shape)

    with open('Y_predicted.bin', 'wb') as file:
        array.array('d', max_acc_model(X_test, mask_test).flatten().cpu().tolist()).tofile(file)
    with open('Y.bin', 'wb') as file:
        array.array('d', Y_test.flatten().cpu().tolist()).tofile(file)
    with open('mask.bin', 'wb') as file:
        array.array('d', mask_test.flatten().cpu().tolist()).tofile(file)

if __name__ == "__main__":
    main()
