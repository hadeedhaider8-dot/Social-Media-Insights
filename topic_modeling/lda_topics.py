import pandas as pd
import re

from gensim import corpora
from gensim.models import LdaModel


# ---------------------------------
# File Path
# ---------------------------------
DATA_FILE = "data/processed_social_media_data.csv"


# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("LDA TOPIC MODELING")
print("=" * 60)

print(f"\nDataset loaded: {len(df)} posts")


# ---------------------------------
# Prepare Text
# ---------------------------------
def prepare_tokens(text):
    text = str(text).lower()

    # Remove hashtags symbol
    text = text.replace("#", "")

    # Keep only words
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return words


documents = [
    prepare_tokens(text)
    for text in df["processed_text"].fillna("")
]


# Remove empty documents
documents = [
    doc for doc in documents
    if len(doc) > 0
]


print(f"Documents prepared: {len(documents)}")


# ---------------------------------
# Create Dictionary
# ---------------------------------
dictionary = corpora.Dictionary(documents)

print(f"Unique words: {len(dictionary)}")


# ---------------------------------
# Create Bag of Words
# ---------------------------------
corpus = [
    dictionary.doc2bow(doc)
    for doc in documents
]


print("Bag-of-Words corpus created.")


# ---------------------------------
# Train LDA Model
# ---------------------------------
NUM_TOPICS = 3

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=20
)


# ---------------------------------
# Display Topics
# ---------------------------------
print("\n" + "=" * 60)
print("DISCOVERED TOPICS")
print("=" * 60)

for topic_id, words in lda_model.print_topics(
    num_topics=NUM_TOPICS,
    num_words=5
):
    print(f"\nTopic {topic_id + 1}:")
    print(words)


# ---------------------------------
# Assign Dominant Topic
# ---------------------------------
topic_results = []

for doc_bow in corpus:

    topic_distribution = lda_model[
        doc_bow
    ]

    dominant_topic = max(
        topic_distribution,
        key=lambda x: x[1]
    )[0]

    topic_results.append(
        dominant_topic + 1
    )


# ---------------------------------
# Save Results
# ---------------------------------
result_df = df.iloc[:len(topic_results)].copy()

result_df["topic"] = topic_results

OUTPUT_FILE = "data/topic_modeling_results.csv"

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------
# Topic Distribution
# ---------------------------------
print("\n" + "=" * 60)
print("TOPIC DISTRIBUTION")
print("=" * 60)

print(
    result_df["topic"].value_counts().sort_index()
)


print(
    f"\nTopic modeling results saved to: {OUTPUT_FILE}"
)

print("\nLDA topic modeling completed successfully!")