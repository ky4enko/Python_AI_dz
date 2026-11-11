from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams
from collections import Counter
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "assets" / "corpus" / "textmind_reviews.txt"
reviews = file_path.read_text(encoding="utf-8").splitlines()
reviews = [review.strip() for review in reviews if review.strip()]

df = pd.DataFrame({"text": reviews})
csv_path = BASE_DIR / "feedback.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")
df = pd.read_csv(csv_path)

df["clean_text"] = df["text"].str.lower()
df["clean_text"] = df["clean_text"].str.replace(r"[^a-z\s]", " ", regex = True)
df["clean_text"] = df["clean_text"].str.replace(r"\s+", " ", regex = True).str.strip()
df["word_count"] = df["clean_text"].str.split().str.len()
print(f"Кількість відгуків:{len(df)} \n"
      f"Середня довживна відгуку: {round(df['word_count'].mean(),2)}")
csv_path = BASE_DIR / "feedback.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

nltk.download("punkt_tab")
nltk.download("stopwords")

df["tokens"] = df["clean_text"].apply(word_tokenize)
token_before =df["tokens"].apply(len).sum()


stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    filtered_tokens = []
    for token in tokens:
          if token not in stop_words and len(token) > 3:
                filtered_tokens.append(token)
    return filtered_tokens
df["filtered_tokens"] = df["tokens"].apply(remove_stopwords)
token_after = df["filtered_tokens"].apply(len).sum()
print(f"Токенів до очистки: {token_before }\n"
      f"Токенів після очистки: {token_after}")

all_tokens = []
for tokens in df["filtered_tokens"]:
      for token in tokens:
            all_tokens.append(token)
word_freq = Counter(all_tokens)
top_15_words = word_freq.most_common(15)
for word, count in top_15_words:
      print(f"{word}: {count}")

frequency_df = pd.DataFrame(top_15_words, columns=["Word", "Frequency"])
plt.figure(figsize = (12,6))
plt.bar(frequency_df["Word"], frequency_df["Frequency"])
plt.title("Top 15 Words Frequency")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.savefig(BASE_DIR / "top_15_words_frequency.png", dpi=300)

bigram_freq = Counter()
for tokens in df["filtered_tokens"]:
    bigram_freq.update(bigrams(tokens))
top_10_bigrams = bigram_freq.most_common(10)
bigram_rows = []
for bigram, count in top_10_bigrams:
    bigram_rows.append([" ".join(bigram), count])
bigram_df = pd.DataFrame(bigram_rows, columns = ["Біграма", "Частота"])
print(bigram_df.to_string(index=False))

bigram_df.to_csv(BASE_DIR / "bigram.csv", index=False, encoding="utf-8")
