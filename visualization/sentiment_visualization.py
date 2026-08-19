import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# Load Sentiment Dataset
# ---------------------------------
DATA_FILE = "data/sentiment_analysis.csv"

df = pd.read_csv(DATA_FILE)


# ---------------------------------
# Sentiment Counts
# ---------------------------------
sentiment_counts = df["sentiment"].value_counts()

print("=" * 50)
print("SENTIMENT VISUALIZATION")
print("=" * 50)

print("\nSentiment Distribution:")
print(sentiment_counts)


# ---------------------------------
# Bar Chart
# ---------------------------------
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment",
    order=["Positive", "Neutral", "Negative"]
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")

plt.tight_layout()
plt.show()


# ---------------------------------
# Pie Chart
# ---------------------------------
plt.figure(figsize=(7, 7))

plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Social Media Sentiment Distribution")

plt.tight_layout()
plt.show()


print("\nSentiment visualization completed successfully!")