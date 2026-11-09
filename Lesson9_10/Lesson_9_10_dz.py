# Завдання 1
# Побудуйте модель лінійної регресії на одній ознаці та порахуйте якість:
# Завантажте датасет California Housing за допомогою fetch_california_housing.
# Візьміть тільки одну ознаку MedInc і випадкові 5000 рядків.
# Розділіть дані на навчальну і тестову вибірки в пропорції 80/20 з фіксованим параметром random_state=42 для відтворюваності результатів.
# Навчіть модель LinearRegression.
# Обчисліть коефіцієнт детермінації R² на тестовій вибірці.
# Виведіть результат у таблиці:

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    # df = df.sample(n = 5000, random_state = 42)
    features = ["MedInc"]
    X = df[features]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    result_table = pd.DataFrame({
        "Модель": ["LinearRegression"],
        "Ознака": ["MedInc"],
        "R²": [round(r2_score(y_test, pred), 3)]
    })

    print(result_table)


if __name__ == "__main__":
    main()

# Завдання 2
# Порівняйте якість моделі на одній ознаці та на двох ознаках:
# Використовуйте датасет California Housing (випадкові 5000 рядків).
# Побудуйте дві моделі:
# тільки з ознакою MedInc;
# з ознаками MedInc і AveRooms.
# Розділіть дані на навчальну і тестову вибірки в пропорції 80/20 з фіксованим параметром random_state=42 для відтворюваності результатів.
# Обчисліть R² на тестовій вибірці.
# Виведіть результати в таблиці:

# def main() -> None:
#     data = fetch_california_housing(as_frame=True)
#     df = data.frame
#     df = df.sample(n = 5000, random_state = 42)
#     X1 = df[["MedInc"]]
#     X2 = df[["MedInc" , "AveRooms"]]
#     y = df["MedHouseVal"]
#
#     X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
#     model = LinearRegression().fit(X1_train, y_train)
#     pred = model.predict(X1_test)
#     r2_first = round(r2_score(y_test, pred), 3)
#
#     X2_train, X2_test, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=42)
#     model = LinearRegression().fit(X2_train, y_train2)
#     pred = model.predict(X2_test)
#     r2_second = round(r2_score(y_test2, pred), 3)
#
#     result_table = pd.DataFrame({
#         "Модель": ["LinearRegression First", "LinearRegression Second"],
#         "Ознака": ["MedInc" , "MedInc + AveRooms"],
#         "R²": [r2_first,r2_second]})
#
#     print(result_table)
#
#
# if __name__ == "__main__":
#     main()

# Завдання 3
# Обчисліть кілька метрик помилки для однієї моделі:
# Використовуйте датасет California Housing (випадкові 5000 рядків).
# Розділіть дані на навчальну і тестову вибірки в пропорції 80/20 з фіксованим параметром random_state=42 для відтворюваності результатів.
# Побудуйте модель лінійної регресії на ознаках MedInc і AveRooms.
# На тестовій вибірці порахуйте:
# MAE (mean_absolute_error).
# MSE (mean_squared_error).
# RMSE як np.sqrt(MSE).
# R² (r2_score).
# Оформіть результати в таблиці:


# def main() -> None:
#     data = fetch_california_housing(as_frame=True)
#     df =data.frame
#     df = df.sample(n = 5000, random_state = 42)
#
#     X = df[["MedInc", "AveRooms"]]
#     y= df["MedHouseVal"]
#
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     y_pred = LinearRegression().fit(x_train, y_train).predict(x_test)
#
#     MAE = round(mean_absolute_error(y_test, y_pred), 3)
#     MSE = round(mean_squared_error(y_test, y_pred), 3)
#     R2 = round(r2_score(y_test, y_pred), 3)
#     RMSE = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
#
#     result_table = pd.DataFrame({
#         "Метрика" : ["MAE", "MSE", "RMSE", "R²"],
#         "Значення" : [MAE , MSE ,RMSE , R2],
#     })
#
#     print(result_table)
#
# if __name__ == "__main__":
#     main()


# Завдання 4
# Побудуйте графік для порівняння реальних і передбачених значень:
# Використовуйте датасет California Housing (випадкові 5000 рядків).
# Розділіть дані на навчальну і тестову вибірки в пропорції 80/20 з фіксованим параметром random_state=42 для відтворюваності результатів.
# Побудуйте модель лінійної регресії на ознаках MedInc і AveRooms.
# Отримайте передбачення на тестовій вибірці.
# Побудуйте scatter-графік: по осі X — y_test, по осі Y — y_pred.
# Додайте лінію y = x (ідеальні передбачення).
# Підпишіть осі та збережіть графік у файл.

def main() -> None:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df = df.sample(n = 5000 , random_state = 42)

    X = df[["MedInc", "AveRooms"]]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = LinearRegression().fit(X_train,y_train)
    pred = model.predict(X_test)

    plt.scatter(y_test, pred, color = "green", label = "Передбачення моделі")

    plt.xlabel("y_test")
    plt.ylabel("y_pred")

    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color = "red",
             label = "Ідеальне передбачення")

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
