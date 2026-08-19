import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# Load sentiment analysis data
# -----------------------------------------
INPUT_FILE = "data/sentiment_analysis.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("SENTIMENT TREND ANALYSIS")
print("=" * 60)

print(f"\nPosts loaded: {len(df)}")

# -----------------------------------------
# Convert date column
# -----------------------------------------
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

# -----------------------------------------
# Sentiment count by date
# -----------------------------------------
sentiment_trend = (
    df.groupby(["date", "sentiment"])
    .size()
    .unstack(fill_value=0)
)

print("\nSentiment Trend:")
print(sentiment_trend)

# -----------------------------------------
# Graph 1: Sentiment Trend
# -----------------------------------------
plt.figure(figsize=(12, 6))

for sentiment in sentiment_trend.columns:
    plt.plot(
        sentiment_trend.index,
        sentiment_trend[sentiment],
        marker="o",
        label=sentiment
    )

plt.title("Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Posts")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------------
# Daily Average Sentiment Score
# -----------------------------------------
average_sentiment = (
    df.groupby("date")["compound_score"]
    .mean()
)

print("\nAverage Sentiment Score by Date:")
print(average_sentiment)

# -----------------------------------------
# Graph 2: Average Sentiment Score
# -----------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    average_sentiment.index,
    average_sentiment.values,
    marker="o"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Average Sentiment Score Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------------
# Save trend analysis
# -----------------------------------------
trend_output = sentiment_trend.reset_index()

trend_output.to_csv(
    "data/sentiment_trend.csv",
    index=False
)

print("\nTrend analysis saved to:")
print("data/sentiment_trend.csv")

print("\nSentiment trend analysis completed successfully!")