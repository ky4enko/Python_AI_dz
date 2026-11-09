# Бізнес-контекст
# Команда VisionScan AI зіткнулася з проблемами перенавчання і нестабільних градієнтів під час навчання моделі. Вам доручено поліпшити стабільність мережі,
# застосувавши батч-нормалізацію і контроль градієнтів.
# Завдання 1
# Модифікуйте CNN із практики.
# Додайте шар BatchNorm2d після кожного Conv2d.
# Використовуйте Dropout(p=0.3) після активації ReLU.
# Навчіть мережу на тих самих даних 10 епох.

# Завдання 2
# Додайте контроль градієнтів.
# Застосуйте обрізку градієнтів (torch.nn.utils.clip_grad_norm_) з максимальним значенням 1.0.
# Зафіксуйте зміну втрат за епохами.


# Завдання 3
# Побудуйте порівняльний графік:
# Модель без нормалізації.
# Модель із BatchNorm і Dropout.


# Завдання 4
# Оформіть таблицю порівняння:

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. НАЛАШТУВАННЯ
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# MNIST:
# картинка 28×28
# 1 канал (чорно-біла)
transform = transforms.ToTensor()

# ============================================================
# 2. ЗАВАНТАЖЕННЯ MNIST
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# 3. CNN
# ============================================================
class DigitCNNOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 1×28×28 -> 16×28×28
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            # 16×28×28 -> 16×14×14
            nn.MaxPool2d(2),
            # 16×14×14 -> 32×14×14
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            # 32×14×14 -> 32×7×7
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32 * 7 * 7,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                10
            )
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

model_original = DigitCNNOriginal().to(device)
print(model_original)

class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            # 1×28×28 -> 16×28×28
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # 16×28×28 -> 16×14×14
            nn.MaxPool2d(2),

            # 16×14×14 -> 32×14×14
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # 32×14×14 -> 32×7×7
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),

            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.classifier(x)

        return x


model = DigitCNN().to(device)

print(model)

# ============================================================
# 4. LOSS + OPTIMIZER
# ============================================================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_function_original = nn.CrossEntropyLoss()
optimizer_original = torch.optim.Adam(
    model_original.parameters(),
    lr=0.001
)

# ============================================================
# 5. НАВЧАННЯ
# ============================================================

epochs = 10
loss_history_original = []
loss_history_update = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # -------------------------
        # Forward
        # -------------------------

        predictions = model(images)
        loss = loss_function(
            predictions,
            labels
        )

        # -------------------------
        # Backpropagation
        # -------------------------

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {total_loss / len(train_loader):.4f}")
    loss_history_update.append(total_loss / len(train_loader))
    print(loss_history_update)

for epoch in range(epochs):
    model_original.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

            # -------------------------
            # Forward
            # -------------------------

        predictions = model_original(images)
        loss = loss_function_original(
        predictions,
        labels)

            # -------------------------
            # Backpropagation
            # -------------------------

        optimizer_original.zero_grad()
        loss.backward()
        optimizer_original.step()
        total_loss += loss.item()
    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {total_loss / len(train_loader):.4f}")
    loss_history_original.append(total_loss / len(train_loader))
    print(loss_history_original)

# ============================================================
# 6. ПЕРЕВІРКА ТОЧНОСТІ
# ============================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        correct += (
                predicted == labels
        ).sum().item()

        total += labels.size(0)

accuracy_update = correct / total * 100

print()
print(f"Accuracy: {accuracy_update:.2f}%")

model_original.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model_original(images)

        predicted = outputs.argmax(dim=1)

        correct += (
                predicted == labels
        ).sum().item()

        total += labels.size(0)

accuracy_original = correct / total * 100

print()
print(f"Accuracy: {accuracy_original:.2f}%")

# ============================================================
# 7. ЗБЕРІГАЄМО МОДЕЛЬ
# ============================================================

torch.save(
    model.state_dict(),
    "mnist_cnn.pth"
)

print("Model saved: mnist_cnn.pth")

torch.save(
    model_original.state_dict(),
    "mnist_cnn_original.pth"
)

print("Model saved: mnist_cnn_original.pth")

# ============================================================
# 8. ЗНАХОДИМО ПО ОДНІЙ КАРТИНЦІ КОЖНОЇ ЦИФРИ
# ============================================================

found = {}

for image, label in test_dataset:

    number = label

    if number not in found:
        found[number] = image

    if len(found) == 10:
        break

# ============================================================
# 9. РОЗПІЗНАЄМО І ЗБЕРІГАЄМО 10 КАРТИНОК
# ============================================================

model.eval()

print()
print("=" * 40)
print("РОЗПІЗНАВАННЯ")
print("=" * 40)

for real_number in range(10):
    image = found[real_number]

    # -------------------------
    # Зберігаємо PNG
    # -------------------------

    pil_image = transforms.ToPILImage()(image)

    filename = f"digit_{real_number}.png"

    pil_image.save(filename)

    # -------------------------
    # Передаємо CNN
    # -------------------------

    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(
            output,
            dim=1
        )
        predicted_number = probabilities.argmax(
            dim=1
        ).item()
        confidence = probabilities[
                         0,
                         predicted_number
                     ].item() * 100

    print(
        f"{filename}: "
        f"справжня цифра = {real_number}, "
        f"CNN = {predicted_number}, "
        f"впевненість = {confidence:.2f}%"
    )

model_original.eval()

print()
print("=" * 40)
print("РОЗПІЗНАВАННЯ")
print("=" * 40)

for real_number in range(10):
    image = found[real_number]

        # -------------------------
        # Зберігаємо PNG
        # -------------------------

    pil_image = transforms.ToPILImage()(image)

    filename = f"digit_{real_number}.png"

    pil_image.save(filename)

        # -------------------------
        # Передаємо CNN
        # -------------------------

    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_original(input_tensor)
        probabilities = torch.softmax(
                output,
                dim=1
            )
        predicted_number = probabilities.argmax(
                dim=1
            ).item()
        confidence = probabilities[
                             0,
                             predicted_number
                         ].item() * 100

    print(
            f"{filename}: "
            f"справжня цифра = {real_number}, "
            f"CNN = {predicted_number}, "
            f"впевненість = {confidence:.2f}%"
        )

list_epoch =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(list_epoch, loss_history_original,
         label ="Loss History Original",)
plt.plot(list_epoch, loss_history_update,
         label ="Loss History Update",)
plt.xlabel("Епохи")
plt.ylabel("Loss History")
plt.title("Lesson 23 24")
plt.legend()
plt.show()


columns = ["Конфігурація", "Accuracy", "Loss", "Примітка"]
data = [
    ["Без нормалізації",
     f"{accuracy_original:.2f}%",
     f"{loss_history_original[-1]:.4f}",
     "Вихідна модель"],

    ["BatchNorm + Dropout + Gradient Clipping",
     f"{accuracy_update:.2f}%",
     f"{loss_history_update[-1]:.4f}",
     "Підвищена точність"]
]

df = pd.DataFrame(columns = columns, data = data)
print(df)


#Завдання 1результаты
# Epoch 1/10, Loss: 0.1857
# Epoch 2/10, Loss: 0.0823
# Epoch 3/10, Loss: 0.0665
# Epoch 4/10, Loss: 0.0597
# Epoch 5/10, Loss: 0.0490
# Epoch 6/10, Loss: 0.0467
# Epoch 7/10, Loss: 0.0437
# Epoch 8/10, Loss: 0.0388
# Epoch 9/10, Loss: 0.0348
# Epoch 10/10, Loss: 0.0351
#
# Accuracy: 99.17%
# Model saved: mnist_cnn.pth
#
# ========================================
# РОЗПІЗНАВАННЯ
# ========================================
# digit_0.png: справжня цифра = 0, CNN = 0, впевненість = 100.00%
# digit_1.png: справжня цифра = 1, CNN = 1, впевненість = 99.76%
# digit_2.png: справжня цифра = 2, CNN = 2, впевненість = 100.00%
# digit_3.png: справжня цифра = 3, CNN = 5, впевненість = 58.89%
# digit_4.png: справжня цифра = 4, CNN = 4, впевненість = 99.78%
# digit_5.png: справжня цифра = 5, CNN = 5, впевненість = 98.83%
# digit_6.png: справжня цифра = 6, CNN = 6, впевненість = 99.95%
# digit_7.png: справжня цифра = 7, CNN = 7, впевненість = 99.95%
# digit_8.png: справжня цифра = 8, CNN = 8, впевненість = 99.96%
# digit_9.png: справжня цифра = 9, CNN = 9, впевненість = 99.99%


# завдання 2 лосс хистори
# Epoch 1/10, Loss: 0.1736
# [0.17364655112200328]
# Epoch 2/10, Loss: 0.0785
# [0.17364655112200328, 0.07849309546066258]
# Epoch 3/10, Loss: 0.0612
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486]
# Epoch 4/10, Loss: 0.0528
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536]
# Epoch 5/10, Loss: 0.0474
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536, 0.04740630903778705]
# Epoch 6/10, Loss: 0.0416
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536, 0.04740630903778705, 0.04164317738381649]
# Epoch 7/10, Loss: 0.0381
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536, 0.04740630903778705, 0.04164317738381649, 0.03814157019824168]
# Epoch 8/10, Loss: 0.0365
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536, 0.04740630903778705, 0.04164317738381649, 0.03814157019824168, 0.03651938707320648]
# Epoch 9/10, Loss: 0.0335
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536, 0.04740630903778705, 0.04164317738381649, 0.03814157019824168, 0.03651938707320648, 0.03350408752308302]
# Epoch 10/10, Loss: 0.0296
# [0.17364655112200328, 0.07849309546066258, 0.061234933552086486, 0.052812545543607536, 0.04740630903778705, 0.04164317738381649, 0.03814157019824168, 0.03651938707320648, 0.03350408752308302, 0.02962910905546008]
#
# Accuracy: 99.22%
# Model saved: mnist_cnn.pth