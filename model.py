import torchvision.models as models
import torch.nn as nn


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.model = models.resnet50(pretrained=True)  # Use ResNet50-pretrained 

        # Unfreezed last 3 layers for fine-tuning
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = FaceRecognitionModel(num_classes).to(device)

# Applied xavier weight initialization for better accuracy
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)

# Set the AdamW optimizer and Cosine annealing scheduler for the end of training if metrics dip
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Model training
def train_model(model, train_loader, test_loader, epochs=30, patience=5):
    model.train()
    best_acc = 0
    early_stop_count = 0
    train_acc_list, test_acc_list = [], []

    for epoch in range(epochs):
        correct, total, train_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)

        # Evaluate model
        test_acc = evaluate_model(model, test_loader)
        test_acc_list.append(test_acc)

        scheduler.step()  # Adjust learning rate
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            early_stop_count = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered!")
                break

    return train_acc_list, test_acc_list

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

train_acc_list, test_acc_list = train_model(model, train_loader, test_loader, epochs=12)

