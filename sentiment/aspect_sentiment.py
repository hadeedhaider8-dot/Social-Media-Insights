import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------
# File Paths
# ---------------------------------
INPUT_FILE = "data/sentiment_analysis.csv"
OUTPUT_FILE = "data/aspect_sentiment.csv"


# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ASPECT-BASED SENTIMENT ANALYSIS")
print("=" * 60)

print(f"\nDataset loaded: {len(df)} posts")


# ---------------------------------
# VADER Analyzer
# ---------------------------------
analyzer = SentimentIntensityAnalyzer()


# ---------------------------------
# Define Common Aspects
# ---------------------------------
aspects = [
    "product",
    "battery",
    "service",
    "customer service",
    "support",
    "software",
    "technology",
    "update",
    "interface",
    "performance",
    "design",
    "quality",
    "features",
    "AI",
    "problem",
]


# ---------------------------------
# Function for Aspect Sentiment
# ---------------------------------
def get_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


aspect_results = []


# ---------------------------------
# Analyze Each Post
# ---------------------------------
for _, row in df.iterrows():

    text = str(row["text"])

    text_lower = text.lower()

    for aspect in aspects:

        if aspect.lower() in text_lower:

            score = analyzer.polarity_scores(text)["compound"]

            sentiment = get_sentiment(score)

            aspect_results.append({
                "post_id": row["post_id"],
                "aspect": aspect,
                "text": text,
                "sentiment_score": score,
                "sentiment": sentiment
            })


# ---------------------------------
# Create Results DataFrame
# ---------------------------------
aspect_df = pd.DataFrame(aspect_results)


# ---------------------------------
# Display Results
# ---------------------------------
print("\n" + "=" * 60)
print("ASPECT SENTIMENT RESULTS")
print("=" * 60)

if len(aspect_df) > 0:

    print(
        aspect_df[
            [
                "aspect",
                "sentiment_score",
                "sentiment"
            ]
        ].to_string(index=False)
    )

else:

    print("No aspects found.")


# ---------------------------------
# Aspect Summary
# ---------------------------------
if len(aspect_df) > 0:

    print("\n" + "=" * 60)
    print("ASPECT SENTIMENT SUMMARY")
    print("=" * 60)

    summary = (
        aspect_df
        .groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    print(summary)


# ---------------------------------
# Save Results
# ---------------------------------
aspect_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nAspect sentiment results saved to: {OUTPUT_FILE}"
)

print("\nAspect-based sentiment analysis completed successfully!")