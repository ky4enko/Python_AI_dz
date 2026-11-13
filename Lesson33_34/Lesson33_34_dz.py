import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

X= df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_all_pca = np.vstack((X_train_pca, X_test_pca))
y_all = np.concatenate((y_train, y_test))

plt.figure(figsize=(8,6))
plt.scatter(X_all_pca[:, 0],
            X_all_pca[:, 1],
            c=y_all,
            cmap="viridis",
            edgecolors="black")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("PCA Components Visualization")
plt.colorbar(label="Wine Class")
plt.grid()
plt.show()

components_list = [2, 5, 10]
accuracies = []
for component in components_list:
    pca = PCA(n_components=component)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Components: {component}, Accuracy: {accuracy:.4f}")

plt.figure(figsize=(8,5))
plt.plot(components_list, accuracies,
         marker="o",linestyle="-")
plt.xlabel("Number components")
plt.ylabel("Accuracy")
plt.title("PCA Accuracy vs Components")
plt.xticks(components_list)
plt.grid(True)
plt.savefig("pca_accuracy_results.png", dpi=300, bbox_inches="tight")
plt.show()