import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise.model_selection import train_test_split

books_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
ratings_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"

books = pd.read_csv(books_url)
ratings = pd.read_csv(ratings_url)
books = books[["book_id", "title", "authors"]]
ratings = ratings[["book_id", "user_id", "rating"]]

books = books.dropna()
ratings = ratings.dropna()
books = books.drop_duplicates(subset=["book_id"])
ratings = ratings.drop_duplicates(subset=["user_id", "book_id"])

df = pd.merge(books, ratings, on="book_id", how="inner")
df = df[["book_id", "title", "authors", "user_id", "rating"]]

books_unique = df[["book_id", "title", "authors"]].drop_duplicates(subset=["book_id"]).reset_index(drop=True)
books_unique["text"] = books_unique["title"] + " " + books_unique["authors"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(books_unique["text"])

similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

def recommend(title, n=5):
    matches = books_unique.index[["title"].str.casefold() == title.casefold()]
    if len(matches) == 0:
        raise ValueError("Book not found")
    book_idx = matches[0]
    scores = similarity_matrix[book_idx].toarray().ravel()
    scores[book_idx] = -1
    n = min(n, len(books_unique) - 1)

    top_indices = np.argpartition(scores, -n)[-n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    result = books_unique.iloc[top_indices][["book_id", "title", "authors"]].copy()
    result["similarity"] = scores[top_indices]
    return result

recommendations = []

for idx, book in books_unique.iterrows():
    scores= similarity_matrix[idx].toarray().ravel()
    scores[idx] = -1
    top_indices = np.argpartition(scores, -5)[-5:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    for rec_idx in top_indices:
        recommendations.append({"book_id": book["book_id"], "title": book["title"],
            "recommended_book_id": books_unique.iloc[rec_idx]["book_id"],
            "recommended_title": books_unique.iloc[rec_idx]["title"],
            "recommended_authors": books_unique.iloc[rec_idx]["authors"],
            "similarity": scores[rec_idx]})
content_recommendations = pd.DataFrame(recommendations)
content_recommendations.to_csv("content_recommendations.csv", index=False, encoding="utf-8-sig")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
trainset , testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD(n_factors=50, n_epochs=10, random_state=42)
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print("RMSE:", rmse)
dump.dump("svd_model.pkl", algo=model)


all_book_ids = [trainset.to_raw_iid(iid)
    for iid in trainset.all_items()]

books_info = books_unique.copy()
books_info["book_id"] = books_info["book_id"].astype(str)
books_info = books_info.set_index("book_id")


# Получаем список всех книг из обученной модели
all_book_ids = [
    trainset.to_raw_iid(iid)
    for iid in trainset.all_items()
]

# Подготавливаем информацию о книгах
books_info = books_unique.set_index("book_id")


def recommend_cf(user_id, n=5):
    try:
        trainset.to_inner_uid(user_id)
    except ValueError:
        raise ValueError("Пользователь не найден в обучающей выборке")

    rated_books = set(df.loc[df["user_id"] == user_id, "book_id"])
    candidates = [book_id for book_id in all_book_ids
        if book_id not in rated_books]
    predictions = [model.predict(user_id, book_id)
        for book_id in candidates]
    predictions.sort(key=lambda prediction: prediction.est, reverse=True)

    recommendations = []

    for prediction in predictions[:n]:
        book = books_info.loc[prediction.iid]
        recommendations.append({
            "user_id": user_id,
            "book_id": prediction.iid,
            "title": book["title"],
            "authors": book["authors"],
            "predicted_rating": round(prediction.est, 3)})
    return pd.DataFrame(recommendations)

user_ids = [trainset.to_raw_uid(uid)
    for uid in list(trainset.all_users())[:3]]

results = []

for user_id in user_ids:
    recommendations = recommend_cf(user_id, n=5)
    print(f"\nРекомендації для користувача {user_id}: {recommendations.to_string()}")
    results.append(recommendations)

cf_results = pd.concat(results, ignore_index=True)
cf_results.to_csv("cf_results.csv", index=False, encoding="utf-8-sig")
print(f"\nРекомендація для трьох користувачів: {cf_results.to_string()}")
