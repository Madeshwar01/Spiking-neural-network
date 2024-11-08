import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class CSVECGDataset(Dataset):
    def init(self, csv_file, feature_cols, label_col, binary=True):
        self.data = pd.read_csv(csv_file)
        self.features = self.data[feature_cols].values
        self.labels = self.data[label_col].values
        if binary:
            self.labels = np.where(self.labels == 'N', 0, 1)
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    def len(self):
        return len(self.labels)
    def getitem(self, idx):
        return self.features[idx], self.labels[idx]

csv_file = "C:\\Users\\karth\\Downloads\\MIT-BIH Arrhythmia Database.csv\\MIT-BIH Arrhythmia Database.csv"
feature_cols = [f'feature_{i}' for i in range(1, 33)]  # 32 features in the dataset
label_col = 'label'

dataset = CSVECGDataset(csv_file, feature_cols, label_col, binary=True)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

num_inputs = 32  
num_hidden = 100
num_outputs = 2  
num_steps = 32  

class BinarySNN(nn.Module):
    def init(self, num_inputs, num_hidden, num_outputs, num_steps):
        super(BinarySNN, self).init()
        self.num_steps = num_steps
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.95)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):
        spk2_rec, mem2_rec = [], []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinarySNN(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs, num_steps=num_steps).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            spk_rec, mem_rec = model(signals)

            loss_val = torch.zeros((1), dtype=torch.float32, device=device)
            for step in range(num_steps):
                loss_val += criterion(mem_rec[step], labels)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return loss_history

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            spk_rec, mem_rec = model(signals)

            _, predicted = torch.max(mem_rec[-1], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Arrhythmia'], yticklabels=['Normal', 'Arrhythmia'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

num_epochs = 100 
loss_history = train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

model_save_path = "binary_snn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

test(model, test_loader)

plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
