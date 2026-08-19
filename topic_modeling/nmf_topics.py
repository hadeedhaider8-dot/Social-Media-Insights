import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# ---------------------------------
# File Paths
# ---------------------------------
INPUT_FILE = "data/processed_social_media_data.csv"
OUTPUT_FILE = "data/nmf_topic_results.csv"


# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("NMF TOPIC MODELING")
print("=" * 60)

print(f"\nDataset loaded: {len(df)} posts")


# ---------------------------------
# Prepare Text
# ---------------------------------
texts = df["processed_text"].fillna("")


# ---------------------------------
# TF-IDF Vectorization
# ---------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(texts)

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# ---------------------------------
# NMF Model
# ---------------------------------
NUM_TOPICS = 3

nmf_model = NMF(
    n_components=NUM_TOPICS,
    random_state=42,
    init="nndsvda",
    max_iter=500
)

nmf_matrix = nmf_model.fit_transform(tfidf_matrix)


# ---------------------------------
# Display Topics
# ---------------------------------
feature_names = vectorizer.get_feature_names_out()

print("\n" + "=" * 60)
print("DISCOVERED NMF TOPICS")
print("=" * 60)

for topic_index, topic in enumerate(
    nmf_model.components_
):

    top_indices = topic.argsort()[-5:][::-1]

    top_words = [
        feature_names[i]
        for i in top_indices
    ]

    print(
        f"\nTopic {topic_index + 1}:"
    )

    print(", ".join(top_words))


# ---------------------------------
# Assign Dominant Topic
# ---------------------------------
dominant_topics = nmf_matrix.argmax(axis=1) + 1

df["nmf_topic"] = dominant_topics


# ---------------------------------
# Save Results
# ---------------------------------
df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------
# Topic Distribution
# ---------------------------------
print("\n" + "=" * 60)
print("NMF TOPIC DISTRIBUTION")
print("=" * 60)

print(
    df["nmf_topic"]
    .value_counts()
    .sort_index()
)


print(
    f"\nNMF results saved to: {OUTPUT_FILE}"
)

print("\nNMF topic modeling completed successfully!")