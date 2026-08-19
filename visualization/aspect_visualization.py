import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load aspect sentiment data
df = pd.read_csv("data/aspect_sentiment.csv")

print("=" * 60)
print("ASPECT SENTIMENT VISUALIZATION")
print("=" * 60)

print(f"\nRecords loaded: {len(df)}")

# Create summary
summary = (
    df.groupby(["aspect", "sentiment"])
    .size()
    .unstack(fill_value=0)
)

print("\nAspect Sentiment Summary:")
print(summary)

# -----------------------------
# Graph 1: Sentiment by Aspect
# -----------------------------
summary.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Aspect-Based Sentiment Distribution")
plt.xlabel("Aspect")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()

# -----------------------------
# Average Sentiment Score
# -----------------------------
average_scores = (
    df.groupby("aspect")["sentiment_score"]
    .mean()
    .sort_values()
)

print("\nAverage Sentiment Score by Aspect:")
print(average_scores)

# -----------------------------
# Graph 2: Average Scores
# -----------------------------
plt.figure(figsize=(12, 6))

sns.barplot(
    x=average_scores.index,
    y=average_scores.values
)

plt.title("Average Sentiment Score by Aspect")
plt.xlabel("Aspect")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45, ha="right")
plt.axhline(y=0, linestyle="--")
plt.tight_layout()
plt.show()

print("\nAspect sentiment visualization completed successfully!")