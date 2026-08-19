import pandas as pd
import os

print("=" * 60)
print("EXPORTING SOCIAL MEDIA ANALYSIS REPORT")
print("=" * 60)

# --------------------------------------------------
# Load all analysis results
# --------------------------------------------------

sentiment_df = pd.read_csv(
    "data/sentiment_analysis.csv"
)

aspect_df = pd.read_csv(
    "data/aspect_sentiment.csv"
)

topic_df = pd.read_csv(
    "data/topic_modeling_results.csv"
)

trend_df = pd.read_csv(
    "data/sentiment_trend.csv"
)

nmf_df = pd.read_csv(
    "data/nmf_topic_results.csv"
)

print("\nAll analysis files loaded successfully.")

# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

output_file = "reports/social_media_analysis_report.xlsx"

# --------------------------------------------------
# Create Excel report
# --------------------------------------------------

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    # Sentiment Analysis
    sentiment_df.to_excel(
        writer,
        sheet_name="Sentiment",
        index=False
    )

    # Aspect Sentiment
    aspect_df.to_excel(
        writer,
        sheet_name="Aspect Sentiment",
        index=False
    )

    # LDA Topics
    topic_df.to_excel(
        writer,
        sheet_name="LDA Topics",
        index=False
    )

    # NMF Topics
    nmf_df.to_excel(
        writer,
        sheet_name="NMF Topics",
        index=False
    )

    # Sentiment Trend
    trend_df.to_excel(
        writer,
        sheet_name="Sentiment Trend",
        index=False
    )

# --------------------------------------------------
# Completion message
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXCEL REPORT CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nReport saved to:")
print(output_file)

print("\nIncluded sheets:")
print("1. Sentiment")
print("2. Aspect Sentiment")
print("3. LDA Topics")
print("4. NMF Topics")
print("5. Sentiment Trend")

print("\nExport and reporting completed successfully!")