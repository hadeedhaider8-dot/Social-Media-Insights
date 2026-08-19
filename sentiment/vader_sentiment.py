import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------
# File Paths
# ---------------------------------
INPUT_FILE = "data/processed_social_media_data.csv"
OUTPUT_FILE = "data/sentiment_analysis.csv"


# ---------------------------------
# Initialize VADER
# ---------------------------------
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze sentiment using VADER.
    Returns sentiment scores and label.
    """

    scores = analyzer.polarity_scores(str(text))

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"

    elif compound <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return pd.Series({
        "positive_score": scores["pos"],
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "compound_score": compound,
        "sentiment": sentiment
    })


def main():

    # ---------------------------------
    # Load Processed Dataset
    # ---------------------------------
    df = pd.read_csv(INPUT_FILE)

    print("=" * 50)
    print("VADER SENTIMENT ANALYSIS")
    print("=" * 50)

    print(f"\nDataset loaded: {len(df)} posts")

    # ---------------------------------
    # Apply Sentiment Analysis
    # ---------------------------------
    sentiment_results = df["text"].apply(analyze_sentiment)

    # Add results to dataset
    df = pd.concat(
        [df, sentiment_results],
        axis=1
    )

    # ---------------------------------
    # Save Results
    # ---------------------------------
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------
    # Display Results
    # ---------------------------------
    print("\nSentiment Results:")

    print(
        df[
            [
                "text",
                "compound_score",
                "sentiment"
            ]
        ].to_string(index=False)
    )

    print("\nSentiment Distribution:")

    print(
        df["sentiment"].value_counts()
    )

    print(
        f"\nSentiment analysis saved to: {OUTPUT_FILE}"
    )

    print("\nVADER sentiment analysis completed successfully!")


if __name__ == "__main__":
    main()