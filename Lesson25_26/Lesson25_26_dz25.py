import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix)



class XrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        label = int(self.labels[index][0])
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    dataset_path = Path(r"E:\PythonAI\PythonAI_dz\Lesson25_26\xray\pneumoniamnist_64.npz")
    data = np.load(dataset_path)

    train_images = data["train_images"]
    train_labels = data["train_labels"]
    val_images = data["val_images"]
    val_labels = data["val_labels"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = XrayDataset(train_images, train_labels, transform=transform)
    val_dataset = XrayDataset(val_images, val_labels, transform=transform)
    test_dataset = XrayDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(
        weights = torchvision.models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=0.0001,
        weight_decay=0.0001
    )
    for epoch in range(5):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item() * images.size(0)
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)

        print(f"Epoch: {epoch}, "
              f"Train Loss: {train_loss},"
              f" Val Loss: {val_loss}, "
              f"Val Accuracy: {val_accuracy}")
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predictions = torch.argmax(output, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f},"
          f"\nPrecision: {precision:.4f},"
          f"\nRecall: {recall:.4f},"
          f"\nF1: {f1:.4f}")

    print(f"Confusion Matrix \n{confusion_matrix(all_labels, all_predictions)}")

    output_dir = Path(r"E:\PythonAI\PythonAI_dz\Lesson25_26")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pneumonia_resnet18.pth"
    torch.save(model.state_dict(),model_path)

    model.eval()
    class_name = ["NORMAL", "PNEUMONIA"]
    indices = torch.randperm(len(test_dataset))[:8].tolist()
    fig, axes = plt.subplots(2, 4 , figsize=(14, 8))

    with torch.no_grad():
        for ax, index in zip(axes.flat, indices):
            image, label = test_dataset[index]
            output = model(image.unsqueeze(0).to(device))
            prediction = torch.argmax(output, dim=1).item()
            display_image = image[0] * 0.229 + 0.485
            ax.imshow(display_image.numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"Actual: {class_name[label]}\n"
                f"Predicted: {class_name[prediction]}")
            ax.axis("off")

    plt.tight_layout()
    prediction_path = output_dir / "pneumonia_predictions.png"
    plt.savefig(prediction_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Predictions saved: {prediction_path}")

if __name__ == "__main__":
    main()



