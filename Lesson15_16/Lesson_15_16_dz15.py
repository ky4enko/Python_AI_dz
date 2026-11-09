# Бізнес-контекст
# Компанія HealthRisk Analytics розширює прототип моделі прогнозування. Тепер потрібно класифікувати клієнтів за
# ймовірністю звернення по медичну допомогу (низький/високий ризик) на підставі демографічних і медичних показників.
# Ви маєте побудувати багатошарову нейромережу (MLP) для бінарної класифікації, оцінити її точність і візуалізувати процес навчання.
#
# Завдання 1
# Підготуйте дані для моделі.
# Використовуйте датасет Heart Disease UCI (або аналогічний CSV з ознаками пацієнтів).
# Розділіть вибірку на train/test у пропорції 80/20 (random_state=42).
# Масштабуйте ознаки через StandardScaler.
# Перевірте баланс класів (низький/високий ризик) і за необхідності застосуйте stratify.

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt



def main() -> None:
    heart = fetch_ucirepo(id=45)

    X = heart.data.features
    y = heart.data.targets

    y = (y["num"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    # print("Train:", X_train.shape, y_train.shape)
    # print("Test:", X_test.shape, y_test.shape)
    #
    # print(X_train.shape)
    # print(X_test.shape)
    #
    # print(X_train.shape, X_train.dtype)
    # print(y_train.shape, y_train.dtype)

    model = nn.Sequential(
        nn.Linear(13 , 16),
        nn.ReLU(),
        nn.Linear(16 , 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid())

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        model.train()

        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted_classes = (model(X_train) >= 0.5).float()
            accuracy = (predicted_classes == y_train).float().mean()

        loss_history.append(loss.item())
        accuracy_history.append(accuracy.item())

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Accuracy: {accuracy.item():.4f}"
        )

    model.eval()

    with torch.no_grad():
        predictions = model(X_test)
        predicted_classes = (predictions >= 0.5).int().numpy().flatten()

    y_true = y_test.numpy().flatten().astype(int)
    print("Test Accuracy:", accuracy_score(y_true, predicted_classes))
    print(classification_report(y_true, predicted_classes))
    print(confusion_matrix(y_true, predicted_classes))


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), loss_history)
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracy_history)
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
