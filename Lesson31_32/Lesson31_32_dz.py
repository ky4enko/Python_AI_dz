import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from pathlib import Path
import dimred
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

df = X.copy()
df['target'] = data.target

print(f"Розмір усієї таблиці:, {df.shape} \n"
      f"Розмір матриці ознак:, {X.shape}")
print("\nКількість об'єктів кожного класу:")
class_counts = df["target"].value_counts().sort_index()
for class_id, count in class_counts.items():
    class_name = data.target_names[class_id]
    print(f"{class_name} ({class_id}): {count}")

std_values = X.std().sort_values()
print(f"\nТри ознаки з найменшим стандартним відхиленням: {std_values.head(3)}"
      f"\n\nТри ознаки з найбільшим стандартним відхиленням:{std_values.tail(3).sort_values(ascending=False)}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

assert np.allclose(X_scaled_df.mean(), 0 , atol=1e-7)
assert np.allclose(X_scaled_df.std(ddof =0),1 ,  atol=1e-7)
assert "target" not in X_scaled_df.columns

svd_scaled = TruncatedSVD(n_components=2, random_state=42)
X_svd_scaled = svd_scaled.fit_transform(X_scaled)
svd_original = TruncatedSVD(n_components=2, random_state=42)
X_svd_original = svd_original.fit_transform(X)

fig, axes = plt.subplots(1 , 2 , figsize=(14,6))
datasets = [(X_svd_scaled, "SVD Projection (зі StandardScaler)"),
            (X_svd_original, "SVD Projection (без масштабування)")]
for ax,  (coordinates, title) in zip (axes, datasets):

    for class_id , class_name in enumerate(data.target_names):
        mask = data.target == class_id
        ax.scatter(coordinates[mask, 0],
                   coordinates[mask, 1],
                   label = class_name,
                   alpha = 0.7,
                   s=25)
    ax.set_title(title)
    ax.set_xlabel("SVD-1")
    ax.set_ylabel("SVD-2")
    ax.legend()
    ax.grid(alpha=0.2)
plt.tight_layout()

output_path = Path(__file__).with_name("svd_projection.png")
plt.savefig(output_path , dpi=300, bbox_inches='tight')
plt.show()

knn_scaled = dimred.knn_score(X_svd_scaled,data.target)
knn_original = dimred.knn_score(X_svd_original,data.target)

print(f"KNN StandardScaler: {knn_scaled:.3f} \nKNN original: {knn_original:.3f}")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\nЧастка дисперсії компонент: {pca.explained_variance_ratio_}"
      f"\nСумарна частка: {pca.explained_variance_ratio_.sum()}")

plt.figure(figsize=[8,6])
plt.scatter(X_pca[:,0],X_pca[:,1],
            c=data.target,
            cmap= "coolwarm",
            alpha = 0.7,)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсії)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсії)")
plt.title("PCA — зменшення розмірності")
plt.colorbar(label="Клас пухлини")
plt.savefig("pca_projection.png", dpi=300, bbox_inches="tight")
plt.show()

for i , components in enumerate(pca.components_, start=1):
    weights= pd.Series(components, index= X.columns)

    print(f"\nТоп-3 ознаки для PC{i}: {weights.abs().sort_values(ascending=False).head(3)}")

X_pca_aligned = dimred.align_signs(X_svd_scaled, X_pca)
max_diff = np.max(np.abs(X_svd_scaled - X_pca_aligned))
print(f"\nМаксимальна різниця координат: {max_diff:.10f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_svd_scaled[:, 0], X_svd_scaled[:, 1],
    c=data.target,
    cmap="coolwarm",
    alpha=0.7)
axes[0].set_title("SVD")
axes[0].set_xlabel("SVD 1")
axes[0].set_ylabel("SVD 2")

axes[1].scatter(X_pca_aligned[:, 0], X_pca_aligned[:, 1],
    c=data.target,
    cmap="coolwarm",
    alpha=0.7)
axes[1].set_title("PCA")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

fig.suptitle("Порівняння SVD та PCA")
plt.tight_layout()
plt.savefig("dimensionality_reduction_comparison.png", dpi=300)
plt.show()

svd_variance = svd_scaled.explained_variance_ratio_.sum()
pca_variance = pca.explained_variance_ratio_.sum()

svd_knn = dimred.knn_score(X_svd_scaled, data.target)
pca_knn = dimred.knn_score(X_pca, data.target)

print(f"\nSVD — дисперсія: {svd_variance:.4f}"
      f"\nPCA — дисперсія: {pca_variance:.4f}")
print(f"\n\nSVD — kNN: {svd_knn:.3f}"
      f"\nPCA — kNN: {pca_knn:.3f}")

# PCA для всіх компонент
pca_full = PCA()
pca_full.fit(X_scaled)

cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_90 = np.argmax(cumulative >= 0.90) + 1

print(f"\nКількість компонент для 90% дисперсії: {n_90}"
      f"\n\nЗбережена дисперсія: {cumulative[n_90 - 1]:.4f}")

components = np.arange(1, len(cumulative) + 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar( components, pca_full.explained_variance_ratio_,
    alpha=0.6,
    label="Частка дисперсії")
ax.plot(components, cumulative,
    marker="o",
    label="Накопичена дисперсія")
ax.axhline(0.90, linestyle="--", label="90% дисперсії")
ax.axvline(n_90, linestyle="--", label=f"{n_90} компонент")
ax.set_xlabel("Кількість компонент")
ax.set_ylabel("Частка дисперсії")
ax.set_title("Scree Plot — PCA")
ax.set_xticks(components)
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("pca_scree_plot.png", dpi=300, bbox_inches="tight")
plt.show()


