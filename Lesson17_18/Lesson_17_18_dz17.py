# Регуляризація та аналіз результатів моделі
# Бізнес-контекст
# Компанія HealthRisk Analytics завершує пілотний етап розробки моделі, яка прогнозує ймовірність хронічних захворювань на підставі медичних даних.
# Тепер необхідно перевірити стійкість моделі на реальному наборі даних і оцінити вплив регуляризації на точність і стабільність прогнозів.
#
# Завдання 1
# Підготуйте дані для аналізу:
# Використовуйте датасет про діабет із бібліотеки sklearn.datasets.load_diabetes.
# Розділіть дані на навчальну і тестову вибірки в пропорції 80/20 (параметр random_state=42).
# Масштабуйте ознаки за допомогою StandardScaler.
# Перевірте форму матриць навчальної та тестової вибірок.
#
# Завдання 2
# Побудуйте модель MLP для регресії:
# Архітектура мережі:
# Вхідний шар — кількість нейронів дорівнює числу ознак.
# Перший прихований шар — 64 нейрони, активація ReLU.
# Другий прихований шар — 32 нейрони, активація ReLU.
# Третій прихований шар — 16 нейронів, активація ReLU.
# Вихідний шар — 1 нейрон без активації.
# Функція втрат: MSELoss.
# Оптимізатор: Adam з learning_rate = 0.001.
# Кількість епох: 50.
#
# Завдання 3
# Додайте регуляризацію і порівняйте результати:
# Використовуйте параметр weight_decay=1e-4 в оптимізаторі Adam для ввімкнення L2-регуляризації.
# Проведіть два експерименти:
# Без регуляризації.
# З регуляризацією (L2).
# Для кожної версії моделі обчисліть значення MAE і R² на тестовій вибірці.
# Оформіть результати в таблиці:
# Конфігурація
# MAE
# R²
# Без регуляризації
#
# З регуляризацією (L2)
#
# Завдання 4
# Проаналізуйте результати моделі:
# Побудуйте scatter-графік, що порівнює реальні та передбачені значення цільової змінної:
# По осі X — реальні значення (y_test).
# По осі Y — передбачені значення (y_pred).
# Додайте діагональ y = x для візуального порівняння.
# Підпишіть осі та заголовок графіка.
# Збережіть зображення під іменем diabetes_healthrisk_analysis.png.
# Зробіть короткий аналітичний висновок:
# Як впливає регуляризація на якість моделі?
# Які ознаки, на вашу думку, найсильніше впливають на результат?


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:

    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(10 , 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    model_l2 = nn.Sequential(
        nn.Linear(10 , 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    optimizer_l2 = torch.optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=0.0001)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(1000):
        model.train()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, loss {loss.item():.4f}")

    for epoch in range(1000):
        model_l2.train()

        y_pred_l2 = model_l2(X_train)
        loss_l2 = criterion(y_pred_l2, y_train)

        optimizer_l2.zero_grad()
        loss_l2.backward()
        optimizer_l2.step()

        if (epoch + 1) % 100 == 0:
            print(f"L2 epoch {epoch + 1}, L2 loss {loss_l2.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    mae = mean_absolute_error(y_test.numpy(), y_pred.numpy())
    r2 = r2_score(y_test.numpy(), y_pred.numpy())

    print(f"MAE: {mae:.2f}, \nR2: {r2:.4f}")

    model_l2.eval()
    with torch.no_grad():
        y_pred_l2 = model_l2(X_test)

    mae_l2 = mean_absolute_error(y_test.numpy(), y_pred.numpy())
    r2_l2 = r2_score(y_test.numpy(), y_pred.numpy())

    print(f"L2 MAE: {mae_l2:.2f}, \nL2 R2: {r2_l2:.4f}")

    y_test_np = y_test.numpy().flatten()
    plt.figure(figsize = (10,10))

    plt.scatter(y_test_np, y_pred, label = "Без регуляризації", alpha = 0.7)
    plt.scatter(y_test_np, y_pred_l2, label = "З L2 регуляризацією", alpha = 0.7)

    min_val =min(y_test_np.min(), y_pred.min(), y_pred_l2.min())
    max_val = max(y_test_np.max(), y_pred.max(), y_pred_l2.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'k--',
        label = "Ідеальне передбачення"
    )
    plt.xlabel("Реальні значення")
    plt.ylabel("Передбачені значення")
    plt.title("Diabetes: реальні та передбачені значення")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Lesson_17_18.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()