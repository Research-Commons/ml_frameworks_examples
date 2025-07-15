import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# ------------------------------
# 1. Transforms
# ------------------------------
transform_train = transforms.Compose([
    #-- randomly crop 32x32 with padding
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #-- add color variation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    #-- normalize to [-1, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ------------------------------
# 2. Load the CIFAR-10 dataset
# ------------------------------
trainset = torchvision.datasets.CIFAR10(root='./generated', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./generated', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# ------------------------------
# 3. CNN Model (more optimized than the experimental one)
# ------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #-- conv block 1: 3 → 32
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #-- downsample by 2
            nn.MaxPool2d(2)
        )
        #-- conv block 2: 32 → 64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #-- conv block 3: 64 → 128
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.5)
        #-- flatten to FC layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        #-- output layer for 10 classes
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ------------------------------
# 4. Training and Evaluation
# ------------------------------
def train(model, optimizer, scheduler, criterion, loader, device):
    #-- enable dropout, batchnorm
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    #-- update LR
    scheduler.step()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    #-- disable dropout, batchnorm
    model.eval()
    correct = 0
    total = 0
    #-- no gradients needed
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #-- take argmax
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            #-- count correct preds
            correct += (predicted == labels).sum().item()
    #-- return accuracy
    return 100 * correct / total

def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            #-- store predictions
            all_preds.extend(predicted.cpu().numpy())
            #-- store actual labels
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# ------------------------------
# 5. Initialize Training
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CNN().to(device)
#-- loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#-- SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#-- decay LR every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ------------------------------
# 6. Run Training Loop
# ------------------------------
num_epochs = 30
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    loss = train(model, optimizer, scheduler, criterion, trainloader, device)
    accuracy = evaluate(model, testloader, device)
    train_losses.append(loss)
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Test Accuracy: {accuracy:.2f}%")

# ------------------------------
# 7. Plot Loss and Accuracy Curves
# ------------------------------

#-- loss curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

#-- accuracy curve
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='green')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------
# 8. Confusion Matrix & Report
# ------------------------------
class_names = trainset.classes
y_pred, y_true = collect_predictions(model, testloader, device)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

#-- print precision, recall, f1-score for each class
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))