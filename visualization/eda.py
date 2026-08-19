import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


# ---------------------------------
# Load Processed Dataset
# ---------------------------------
DATA_FILE = "data/processed_social_media_data.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 50)
print("SOCIAL MEDIA DATASET - EDA")
print("=" * 50)


# ---------------------------------
# 1. Dataset Overview
# ---------------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nPlatform Distribution:")
print(df["platform"].value_counts())


# ---------------------------------
# 2. Engagement Statistics
# ---------------------------------
print("\nEngagement Statistics:")

print("\nLikes:")
print(df["likes"].describe())

print("\nComments:")
print(df["comments"].describe())

print("\nShares:")
print(df["shares"].describe())


# ---------------------------------
# 3. Top Hashtags
# ---------------------------------
hashtags = []

for value in df["hashtags"].dropna():

    tags = str(value).split()

    for tag in tags:
        hashtags.append(tag.lower())

hashtag_counts = Counter(hashtags)

print("\nTop Hashtags:")

for hashtag, count in hashtag_counts.most_common(10):
    print(f"{hashtag}: {count}")


# ---------------------------------
# 4. Word Frequency
# ---------------------------------
all_words = []

for text in df["processed_text"].dropna():

    words = str(text).split()

    for word in words:
        all_words.append(word)

word_counts = Counter(all_words)

print("\nTop 15 Words:")

for word, count in word_counts.most_common(15):
    print(f"{word}: {count}")


# ---------------------------------
# 5. Platform Visualization
# ---------------------------------
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="platform"
)

plt.title("Social Media Posts by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Posts")

plt.tight_layout()
plt.show()


# ---------------------------------
# 6. Engagement Visualization
# ---------------------------------
engagement_data = df[
    ["likes", "comments", "shares"]
]

engagement_data.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Social Media Engagement")
plt.xlabel("Post Index")
plt.ylabel("Engagement Count")

plt.tight_layout()
plt.show()


# ---------------------------------
# 7. Top Hashtags Visualization
# ---------------------------------
top_hashtags = hashtag_counts.most_common(10)

if top_hashtags:

    hashtag_names = [item[0] for item in top_hashtags]
    hashtag_values = [item[1] for item in top_hashtags]

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=hashtag_values,
        y=hashtag_names
    )

    plt.title("Top Hashtags")
    plt.xlabel("Frequency")
    plt.ylabel("Hashtag")

    plt.tight_layout()
    plt.show()


# ---------------------------------
# 8. Word Cloud
# ---------------------------------
all_text = " ".join(
    df["processed_text"].dropna().astype(str)
)

if all_text.strip():

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(all_text)

    plt.figure(figsize=(12, 6))

    plt.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    plt.axis("off")

    plt.title("Social Media Word Cloud")

    plt.show()


print("\nEDA completed successfully!")