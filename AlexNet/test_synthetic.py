"""Test training AlexNet with synthetic data."""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from AlexNet.model import AlexNet

# Create synthetic ImageNet-sized data
print("Creating synthetic ImageNet-sized data...")
X_train = torch.randn(100, 3, 227, 227)  # AlexNet input shape
y_train = torch.randint(0, 1000, (100,))  # 1000 classes

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# AlexNet model
model = AlexNet(num_classes=1000).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Training AlexNet on synthetic data...")
for epoch in range(3):
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if batch_idx == 0:
            first_param = list(model.parameters())[0]
            print(f"Epoch {epoch+1}, Batch {batch_idx}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Gradient std: {first_param.grad.std():.6f}")
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}\n")
