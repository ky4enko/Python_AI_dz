# Завдання 1
# Побудуйте модель поліноміальної регресії для однієї ознаки на базі датасету California Housing:
# Завантажте датасет fetch_california_housing.
# Використовуйте тільки одну ознаку MedInc.
# Створіть поліноміальні ознаки ступеня 2 за допомогою PolynomialFeatures.
# Розділіть дані на навчальну і тестову вибірки (80/20, random_state=42).
# Навчіть модель і обчисліть R² на тестовій вибірці.




from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing
import pandas as pd

# def main() -> None:
#
#     data = fetch_california_housing(as_frame=True)
#     df = data.frame
#     X = df[["MedInc"]]
#     y = df["MedHouseVal"]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     degree = 2
#     model = Pipeline([
#         ("poly", PolynomialFeatures(degree=degree)),
#         ("reg", LinearRegression()),
#     ]).fit(X_train, y_train)
#     print(
#             f"degree={degree}",
#             "test", round(r2_score(y_test, model.predict(X_test)), 3),
#         )
#
# if __name__ == "__main__":
#     main()

# Завдання 2
# Перевірте, як ступінь полінома впливає на якість моделі:
# Використовуйте ознаку MedInc з California Housing.
# Побудуйте моделі з PolynomialFeatures ступенів 1, 2 і 3.
# Для кожної моделі обчисліть R² на train і test.
# Запишіть результати в таблицю:



# def main() -> None:
#
#     data = fetch_california_housing(as_frame=True)
#     df = data.frame
#     X = df[["MedInc"]]
#     y = df["MedHouseVal"]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     result = []
#
#     for degree in (1, 2, 3):
#         model = Pipeline([
#             ("poly", PolynomialFeatures(degree=degree)),
#             ("reg", LinearRegression()),
#         ]).fit(X_train, y_train)
#
#         result.append({"Ступінь полінома":degree ,
#                        "R² train": round(r2_score(y_train, model.predict(X_train)), 3),
#                        "R² test": round(r2_score(y_test, model.predict(X_test)), 3)})
#     result_df = pd.DataFrame(result)
#     print(result_df)
#
#
# if __name__ == "__main__":
#     main()

# Завдання 3
# Застосуйте Lasso-регуляризацію і порівняйте зі звичайною лінійною регресією:
# Використовуйте датасет California Housing.
# Візьміть ознаки MedInc, HouseAge, AveRooms, AveBedrms.
# Побудуйте дві моделі:
# LinearRegression;
# Lasso(alpha=0,1).
# Розділіть дані на train/test (80/20, random_state=42).
# Для кожної моделі обчисліть R² на тестовій вибірці.
# Порівняйте коефіцієнти моделей: скільки з них стали рівними 0 у Lasso.


# def main() -> None:
#
#     data = fetch_california_housing(as_frame=True)
#     df = data.frame
#     X = df[["MedInc", "HouseAge", "AveRooms", "AveBedrms" ]]
#     y = df["MedHouseVal"]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     models = {
#         "LinearRegression": LinearRegression(),
#         "Lasso": Lasso(alpha=0.1),
#     }
#
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         zero = int((model.coef_ == 0).sum())
#         print(name, "R²", round(r2_score(y_test, model.predict(X_test)), 3), "zero", zero)
#
#
# if __name__ == "__main__":
#     main()

# Завдання 4
# Застосуйте Ridge-регуляризацію і порівняйте зі звичайною лінійною регресією:
# Використовуйте ті самі ознаки (MedInc, HouseAge, AveRooms, AveBedrms).
# Побудуйте дві моделі:
# LinearRegression;
# Ridge(alpha=1,0).
# Розділіть дані на train/test (80/20, random_state=42).
# Для кожної моделі обчисліть R² на тестовій вибірці.
# Оформіть результати в таблиці:

def main() -> None:

    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df[["MedInc", "HouseAge", "AveRooms", "AveBedrms" ]]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1),
    }

    result= []

    for name, model in models.items():
        model.fit(X_train, y_train)

        result.append({"Модель": name,
                       "R²": round(r2_score(y_test, model.predict(X_test)), 3),
                       "Ненульові коеф." :int((model.coef_ != 0).sum())})

    result_df = pd.DataFrame(result)
    print(result_df)

if __name__ == "__main__":
    main()

