"""Спільний код пар 31–32: дані wine, масштабування, SVD, PCA, графіки, оцінка 2D-карти.

Дані — sklearn.datasets.load_wine: 178 вин, 13 хімічних ознак, 3 сорти. Вони вже лежать усередині
scikit-learn, інтернет не потрібен. Для SVD на тексті — assets/supportflow_qa.csv з пар 29–30.

Шлях даних у кожному скрипті той самий:
    таблиця n × p → StandardScaler → TruncatedSVD або PCA → n × 2 → scatter
Сорт вина (target) у навчанні не бере участі: ним лише розфарбовуємо точки.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

#matplotlib.use("Agg")  # графіки пишемо у файли, вікно не потрібне
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
QA_CSV = HERE / "assets" / "supportflow_qa.csv"
SEED = 42
COLORS = ["#2a9d8f", "#e76f51", "#264653"]


# ---------------------------------------------------------------- дані
def load_wine_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Таблиця ознак (178 × 13) і окремо сорт вина (0, 1, 2)."""
    bunch = load_wine(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()


def scale(features) -> np.ndarray:
    """z-score кожного стовпця: (x − середнє) / стандартне відхилення."""
    return StandardScaler().fit_transform(features)


def load_questions(path: Path = QA_CSV) -> list[dict]:
    """Рядки CSV як словники: id, topic, question, answer."""
    if not path.is_file():
        raise SystemExit(f"Немає {path} — запускайте з папки пакета")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- моделі
def svd_2d(x: np.ndarray, k: int = 2) -> tuple[np.ndarray, TruncatedSVD]:
    """TruncatedSVD: дані НЕ центрує. Повертає нові координати і навчену модель."""
    model = TruncatedSVD(n_components=k, random_state=SEED)
    return model.fit_transform(x), model


def pca_2d(x: np.ndarray, k: int = 2) -> tuple[np.ndarray, PCA]:
    """PCA: спершу віднімає середнє кожного стовпця, потім робить SVD."""
    model = PCA(n_components=k, random_state=SEED)
    return model.fit_transform(x), model


def align_signs(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Знак компоненти довільний: вісь можна розвернути. Розвертає стовпці other так, як у reference."""
    signs = np.sign((reference * other).sum(axis=0))
    signs[signs == 0] = 1
    return other * signs


# ---------------------------------------------------------------- оцінка і графіки
def knn_score(z: np.ndarray, y, neighbors: int = 5) -> float:
    """Наскільки добре сорти розділені на карті: точність kNN (5 сусідів, 5-fold) на координатах z.

    Це перевірка карти, а не частина SVD / PCA: самі алгоритми сорту не бачать.
    """
    return float(cross_val_score(KNeighborsClassifier(neighbors), z, y, cv=5).mean())


def scatter(ax, z: np.ndarray, y, title: str, xlabel: str, ylabel: str) -> None:
    """Точки за двома компонентами, колір — сорт вина."""
    y = np.asarray(y)
    for cls in np.unique(y):
        pts = z[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.85, color=COLORS[int(cls) % len(COLORS)],
                   label=f"сорт {cls}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def save(fig, name: str) -> Path:
    """Зберігає рисунок у results/ ДО будь-якого plt.show() і закриває його."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def rel(path: Path) -> str:
    return str(path.relative_to(HERE))
