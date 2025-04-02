import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import BeehiveDataset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class SimpleBeehiveNet(nn.Module):
    def __init__(self, input_size=13 * 157, hidden_size=64, output_size=1):
        super(SimpleBeehiveNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  
        return x


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for waveforms, mfccs, labels in tqdm(train_loader):
            mfccs, labels = mfccs.to(device), labels.to(device)

            outputs = model(mfccs)
            loss = criterion(outputs, labels.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predicted = (outputs >= 0.5).float() 
            train_correct += (predicted.view(-1) == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for waveforms, mfccs, labels in tqdm(val_loader):
                mfccs, labels = mfccs.to(device), labels.to(device)

                outputs = model(mfccs)
                loss = criterion(outputs, labels.view(-1, 1))
                val_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                val_correct += (predicted.view(-1) == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

if __name__ == "__main__":
    csv_path = "dataset/all_data_updated.csv"
    audio_dir = "dataset/sound_files/sound_files"
    snippet_duration = 5
    num_mfcc = 13
    sample_rate = 16000
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.00025
    hidden_dim = 64
    output_dim = 1 
    train_val_split = 0.8

    dataset = BeehiveDataset(
        csv_path=csv_path,
        audio_dir=audio_dir,
        snippet_duration=snippet_duration,
        num_mfcc=num_mfcc,
        sample_rate=sample_rate,
    )

    train_size = int(len(dataset) * train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = 'cpu'
    model = SimpleBeehiveNet()
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
