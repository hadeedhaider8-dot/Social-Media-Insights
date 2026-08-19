import pandas as pd


# ============================================================
# INSIGHT GENERATION
# ============================================================

print("=" * 60)
print("SOCIAL MEDIA INSIGHTS GENERATOR")
print("=" * 60)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

sentiment_df = pd.read_csv("data/sentiment_analysis.csv")
aspect_df = pd.read_csv("data/aspect_sentiment.csv")
topic_df = pd.read_csv("data/topic_modeling_results.csv")


print(f"\nSentiment posts loaded: {len(sentiment_df)}")
print(f"Aspect records loaded: {len(aspect_df)}")
print(f"Topic records loaded: {len(topic_df)}")


# ============================================================
# 1. Overall Sentiment
# ============================================================

sentiment_counts = sentiment_df["sentiment"].value_counts()

total_posts = len(sentiment_df)

positive_count = sentiment_counts.get("Positive", 0)
negative_count = sentiment_counts.get("Negative", 0)
neutral_count = sentiment_counts.get("Neutral", 0)

positive_percentage = (positive_count / total_posts) * 100
negative_percentage = (negative_count / total_posts) * 100
neutral_percentage = (neutral_count / total_posts) * 100


print("\n" + "=" * 60)
print("OVERALL SENTIMENT")
print("=" * 60)

print(f"Positive: {positive_count} ({positive_percentage:.2f}%)")
print(f"Negative: {negative_count} ({negative_percentage:.2f}%)")
print(f"Neutral:  {neutral_count} ({neutral_percentage:.2f}%)")


# ============================================================
# 2. Overall Brand Sentiment Score
# ============================================================

overall_score = sentiment_df["compound_score"].mean()

print(f"\nOverall Sentiment Score: {overall_score:.4f}")


if overall_score > 0.05:
    overall_label = "Positive"
elif overall_score < -0.05:
    overall_label = "Negative"
else:
    overall_label = "Neutral"


print(f"Overall Sentiment: {overall_label}")


# ============================================================
# 3. Most Positive Aspect
# ============================================================

aspect_scores = (
    aspect_df.groupby("aspect")["sentiment_score"]
    .mean()
    .sort_values(ascending=False)
)

most_positive_aspect = aspect_scores.index[0]
most_positive_score = aspect_scores.iloc[0]

most_negative_aspect = aspect_scores.index[-1]
most_negative_score = aspect_scores.iloc[-1]


print("\n" + "=" * 60)
print("ASPECT INSIGHTS")
print("=" * 60)

print(
    f"Most Positive Aspect: "
    f"{most_positive_aspect} ({most_positive_score:.4f})"
)

print(
    f"Most Negative Aspect: "
    f"{most_negative_aspect} ({most_negative_score:.4f})"
)


# ============================================================
# 4. Most Discussed Aspect
# ============================================================

aspect_frequency = aspect_df["aspect"].value_counts()

most_discussed_aspect = aspect_frequency.index[0]
most_discussed_count = aspect_frequency.iloc[0]


print(
    f"Most Discussed Aspect: "
    f"{most_discussed_aspect} ({most_discussed_count} mentions)"
)


# ============================================================
# 5. Topic Importance
# ============================================================

topic_counts = topic_df["topic"].value_counts()

most_important_topic = topic_counts.index[0]
most_important_topic_count = topic_counts.iloc[0]


print("\n" + "=" * 60)
print("TOPIC INSIGHTS")
print("=" * 60)

print(
    f"Most Important Topic: "
    f"Topic {most_important_topic}"
)

print(
    f"Posts Associated With Topic: "
    f"{most_important_topic_count}"
)


# ============================================================
# 6. Business Recommendations
# ============================================================

recommendations = []


if negative_percentage > positive_percentage:
    recommendations.append(
        "Negative sentiment is higher than positive sentiment. "
        "The organization should investigate customer complaints."
    )
else:
    recommendations.append(
        "Positive sentiment is dominant. "
        "The organization should maintain the current customer experience."
    )


if most_negative_score < 0:
    recommendations.append(
        f"The aspect '{most_negative_aspect}' has the lowest sentiment. "
        f"Improvement should be prioritized in this area."
    )


if most_positive_score > 0.5:
    recommendations.append(
        f"'{most_positive_aspect}' receives strong positive feedback. "
        f"This strength should be maintained."
    )


recommendations.append(
    f"'{most_discussed_aspect}' is the most frequently discussed aspect. "
    f"It should receive close attention in future analysis."
)


# ============================================================
# 7. Display Recommendations
# ============================================================

print("\n" + "=" * 60)
print("ACTIONABLE RECOMMENDATIONS")
print("=" * 60)

for number, recommendation in enumerate(recommendations, start=1):
    print(f"{number}. {recommendation}")


# ============================================================
# 8. Save Insights Report
# ============================================================

report_lines = []

report_lines.append("SOCIAL MEDIA INSIGHTS REPORT")
report_lines.append("=" * 60)

report_lines.append("\nOVERALL SENTIMENT")
report_lines.append(
    f"Positive: {positive_count} ({positive_percentage:.2f}%)"
)
report_lines.append(
    f"Negative: {negative_count} ({negative_percentage:.2f}%)"
)
report_lines.append(
    f"Neutral: {neutral_count} ({neutral_percentage:.2f}%)"
)

report_lines.append(
    f"\nOverall Sentiment Score: {overall_score:.4f}"
)

report_lines.append(
    f"Overall Sentiment: {overall_label}"
)

report_lines.append("\nASPECT INSIGHTS")

report_lines.append(
    f"Most Positive Aspect: "
    f"{most_positive_aspect} ({most_positive_score:.4f})"
)

report_lines.append(
    f"Most Negative Aspect: "
    f"{most_negative_aspect} ({most_negative_score:.4f})"
)

report_lines.append(
    f"Most Discussed Aspect: "
    f"{most_discussed_aspect} ({most_discussed_count} mentions)"
)

report_lines.append("\nTOPIC INSIGHTS")

report_lines.append(
    f"Most Important Topic: Topic {most_important_topic}"
)

report_lines.append(
    f"Posts Associated With Topic: {most_important_topic_count}"
)

report_lines.append("\nACTIONABLE RECOMMENDATIONS")

for number, recommendation in enumerate(recommendations, start=1):
    report_lines.append(
        f"{number}. {recommendation}"
    )


with open(
    "reports/social_media_insights_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(report_lines))


print("\n" + "=" * 60)
print("INSIGHTS REPORT SAVED")
print("=" * 60)

print(
    "reports/social_media_insights_report.txt"
)

print("\nInsight generation completed successfully!")