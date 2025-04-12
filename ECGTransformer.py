import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define Positional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Define ECG Transformer
class ECGTransformer(nn.Module):
    def __init__(self, num_leads=1, seq_length=1000, d_model=32, num_heads=2, num_layers=1, num_classes=2):
        super(ECGTransformer, self).__init__()
        
        # Linear Projection
        self.embedding = nn.Linear(num_leads, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=64, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        x = self.pos_encoding(x)  # Add Positional Encoding
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        x = self.transformer(x)  # Transformer Encoder
        x = x.mean(dim=0)  # Average Pooling
        out = self.fc(x)  # Classification
        return out

num_leads = 15
seq_length = 2000
num_classes = 2
batch_size = 8
epochs = 20

save_dir = 'D:\\OxfordHomework\\CM\\'
X_train = np.load(save_dir + 'X_data_L.npy')
Y_train = np.load(save_dir + 'Y_data_L.npy')
labels_0 = (Y_train == 0)
X_train_0 = X_train[labels_0, : , :]
Y_train_0 = Y_train[labels_0]

labels_1 = (Y_train == 1)
X_train_1 = X_train[labels_1, : , :]
Y_train_1 = Y_train[labels_1]
'''
labels_2 = (Y_train == 2)
X_train_2 = X_train[labels_2, : , :]
Y_train_2 = Y_train[labels_2]

indices = np.random.choice(np.sum(labels_0), np.sum(labels_2), replace=False)
X_train_0 = X_train_0[indices, : , :]
Y_train_0 = Y_train_0[indices]
'''
indices = np.random.choice(np.sum(labels_1), np.sum(labels_0), replace=False)
X_train_1 = X_train_1[indices, : , :]
Y_train_1 = Y_train_1[indices]

X_train = np.concatenate([X_train_0, X_train_1], axis = 0)
Y_train = np.concatenate([Y_train_0, Y_train_1], axis = 0)
print(X_train.shape)
print(Y_train.shape)
Y_train = np.eye(num_classes)[Y_train]

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

dataset = TensorDataset(X_train, Y_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGTransformer(num_leads=num_leads, seq_length=seq_length, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_list = []
accuracy_list = []
precision_list = []
recall_list = []

train_loss_list = []
train_accuracy_list = []
test_loss_list = []

for epoch in range(epochs):
    epoch_list.append(epoch)
    model.train()
    iter = 0
    total_loss = 0

    total_corr = 0
    total_acc = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        y_true = torch.argmax(y_batch, dim=1) 
        preds = torch.argmax(outputs, dim=1) 

        total_corr += (preds == y_true).sum().item()
        total_acc += y_batch.size(0)
        accuracy = total_corr / total_acc

        iter += 1
        if iter % 100 == 0:
            print(f"Iter {iter}, Loss: {loss}, Accuracy: {accuracy}")
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    train_accuracy_list.append(total_corr/total_acc)
    train_loss_list.append(total_loss/len(train_loader))

    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            y_true = torch.argmax(y_batch, dim=1) 
            preds = torch.argmax(outputs, dim=1) 
            correct += (preds == y_true).sum().item()
            total += y_batch.size(0)

            test_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_true.cpu().numpy())

    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')

    accuracy = correct / total

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    test_loss_list.append(test_loss/len(test_loader))

    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

plt.plot(epoch_list, accuracy_list, label = 'accuracy')
plt.plot(epoch_list, precision_list, label = 'precision')
plt.plot(epoch_list, recall_list, label = 'recall')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()

ax1.plot(epoch_list, train_accuracy_list, label = 'Train Accuracy')
ax1.plot(epoch_list, accuracy_list, label = 'Test Accuracy')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()

ax2.plot(epoch_list, train_loss_list, color = 'purple', label = 'Train Loss')
ax2.plot(epoch_list, test_loss_list, color = 'green', label = 'Test Loss')
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()