import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import re
from gensim.models import Word2Vec
from collections import Counter
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
program_start = time.perf_counter()

nltk.download('movie_reviews')
fileids = movie_reviews.fileids()

reviews = [movie_reviews.raw(fileid) for fileid in fileids]
labels = [movie_reviews.categories(fileid)[0] for fileid in fileids]

df = pd.DataFrame({"text": reviews, "label": labels})

stop_words = set(stopwords.words('english'))

def process_review(review):
    review_sentences = sent_tokenize(review)
    result = []

    for sentence in review_sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        tokens = word_tokenize(sentence)
        filtered_tokens = [token for token in tokens
                           if token not in stop_words]
        if filtered_tokens:
            result.append(filtered_tokens)
    return result

sentences = []
for review in df["text"]:
    sentences.extend(process_review(review))

# start_time = time.perf_counter()

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=16,  # 4 = 5.15 | 8 = 3.58  | 12 = 2.73  | 14 = 2.55 | 16 = 2.41
    sg=1,
    epochs=10,
)

# end_time = time.perf_counter()
# training_time = end_time - start_time
# print(f"Час навчання моделі Word2Vec: {training_time:.2f} секунд")

word_counts = Counter( word for sentence in sentences for word in sentence)
key_words = [word for word, count in word_counts.most_common()
             if word in model.wv.key_to_index][:10]
result =[]
for word in key_words:
    similar_words = model.wv.most_similar(word, topn=5)
    for similar_word, similarity in similar_words:
        result.append({
            "Ключове слово:" : word,
            "Близькі слова": similar_word,
            "Косинусна схожість": round(similarity, 3)
        })
results_df = pd.DataFrame(result)
print(results_df.to_string(index=False))

words= model.wv.index_to_key
word_vectors = model.wv.vectors

kmeans = KMeans(n_clusters=5, random_state=42 , n_init=10)
cluster_labels = kmeans.fit_predict(word_vectors)

clusters_df = pd.DataFrame({
    "word": words,
    "cluster": cluster_labels,
})
for cluster_id in range(5):
    cluster_words = clusters_df[clusters_df["cluster"] == cluster_id]["word"].head(10).tolist()
    print(f"\nКластер {cluster_id + 1}:"
          f", ".join(cluster_words))

pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

plt.figure(figsize=(14, 10))
for cluster_id in range(5):
    mask = cluster_labels == cluster_id
    plt.scatter(word_vectors_2d[mask, 0],
                word_vectors_2d[mask, 1],
                s=8,
                alpha=0.8,
                label=f"Cluster {cluster_id + 1}")
for cluster_id in range(5):
    count = 0
    for i , label in enumerate(cluster_labels):
        if label == cluster_id:
            plt.annotate(words[i],
                         (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                         fontsize = 12)
            count += 1
            if count == 3:
                break
plt.title("Семантичні кластери Word2Vec")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.legend(title = "Кластер")
plt.grid(alpha = 0.2)
plt.tight_layout()

output_path = Path(__file__).with_name("semantic_clusters.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

program_end = time.perf_counter()
total_time = program_end - program_start
print(f"Час виконання програми : {total_time:.2f} секунд")