import streamlit as st
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Social Media Insights",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Social Media Insights Dashboard")
st.write(
    "Sentiment Analysis, Topic Modeling and Business Insights "
    "for Social Media Data"
)

st.markdown("---")


# ============================================================
# LOAD DATA
# ============================================================

sentiment_file = "data/sentiment_analysis.csv"
aspect_file = "data/aspect_sentiment.csv"
lda_file = "data/topic_modeling_results.csv"
nmf_file = "data/nmf_topic_results.csv"
trend_file = "data/sentiment_trend.csv"
report_file = "reports/social_media_analysis_report.xlsx"


try:

    sentiment_df = pd.read_csv(sentiment_file)
    aspect_df = pd.read_csv(aspect_file)
    lda_df = pd.read_csv(lda_file)
    nmf_df = pd.read_csv(nmf_file)
    trend_df = pd.read_csv(trend_file)

except FileNotFoundError as error:

    st.error(
        f"Required file not found: {error.filename}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Sentiment Analysis",
        "Aspect Sentiment",
        "Topic Modeling",
        "Trend Analysis",
        "Business Insights",
        "Data & Reports"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("🏠 Project Overview")

    total_posts = len(sentiment_df)

    positive_count = (
        sentiment_df["sentiment"] == "Positive"
    ).sum()

    negative_count = (
        sentiment_df["sentiment"] == "Negative"
    ).sum()

    neutral_count = (
        sentiment_df["sentiment"] == "Neutral"
    ).sum()

    overall_score = sentiment_df[
        "compound_score"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Posts",
        total_posts
    )

    col2.metric(
        "Positive Posts",
        positive_count
    )

    col3.metric(
        "Negative Posts",
        negative_count
    )

    col4.metric(
        "Overall Score",
        round(overall_score, 3)
    )

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        sentiment_df,
        use_container_width=True
    )

    st.info(
        "Use the sidebar to explore sentiment, aspects, "
        "topics, trends and business insights."
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

elif page == "Sentiment Analysis":

    st.header("😊 Sentiment Analysis")

    sentiment_counts = (
        sentiment_df["sentiment"]
        .value_counts()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Positive",
        sentiment_counts.get("Positive", 0)
    )

    col2.metric(
        "Negative",
        sentiment_counts.get("Negative", 0)
    )

    col3.metric(
        "Neutral",
        sentiment_counts.get("Neutral", 0)
    )

    st.markdown("---")

    st.subheader("Sentiment Distribution")

    st.bar_chart(
        sentiment_counts
    )

    st.subheader("Sentiment Scores")

    st.dataframe(
        sentiment_df[
            [
                "text",
                "compound_score",
                "sentiment"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# ASPECT SENTIMENT
# ============================================================

elif page == "Aspect Sentiment":

    st.header("🔍 Aspect-Based Sentiment Analysis")

    aspect_summary = (
        aspect_df
        .groupby(
            ["aspect", "sentiment"]
        )
        .size()
        .unstack(fill_value=0)
    )

    st.subheader(
        "Sentiment Distribution by Aspect"
    )

    st.bar_chart(
        aspect_summary
    )

    st.markdown("---")

    average_scores = (
        aspect_df
        .groupby("aspect")[
            "sentiment_score"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.subheader(
        "Average Sentiment Score by Aspect"
    )

    st.bar_chart(
        average_scores
    )

    st.subheader(
        "Aspect Sentiment Data"
    )

    st.dataframe(
        aspect_df,
        use_container_width=True
    )


# ============================================================
# TOPIC MODELING
# ============================================================

elif page == "Topic Modeling":

    st.header("🧠 Topic Modeling")

    st.subheader("LDA Topic Distribution")

    lda_counts = (
        lda_df["topic"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        lda_counts
    )

    st.subheader("LDA Results")

    st.dataframe(
        lda_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("NMF Topic Distribution")

    nmf_counts = (
        nmf_df["nmf_topic"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        nmf_counts
    )

    st.subheader("NMF Results")

    st.dataframe(
        nmf_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("LDA vs NMF")

    comparison = pd.DataFrame(
        {
            "LDA": lda_counts,
            "NMF": nmf_counts
        }
    ).fillna(0)

    st.bar_chart(
        comparison
    )


# ============================================================
# TREND ANALYSIS
# ============================================================

elif page == "Trend Analysis":

    st.header("📈 Sentiment Trend Analysis")

    trend_df["date"] = pd.to_datetime(
        trend_df["date"]
    )

    trend_df = trend_df.set_index(
        "date"
    )

    st.subheader(
        "Sentiment Trend Over Time"
    )

    sentiment_columns = [
        column
        for column in [
            "Positive",
            "Negative",
            "Neutral"
        ]
        if column in trend_df.columns
    ]

    st.line_chart(
        trend_df[sentiment_columns]
    )

    st.markdown("---")

    st.subheader(
        "Daily Sentiment Activity"
    )

    st.dataframe(
        trend_df,
        use_container_width=True
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.header("💡 Business Insights")

    total_posts = len(
        sentiment_df
    )

    positive = (
        sentiment_df["sentiment"]
        == "Positive"
    ).sum()

    negative = (
        sentiment_df["sentiment"]
        == "Negative"
    ).sum()

    neutral = (
        sentiment_df["sentiment"]
        == "Neutral"
    ).sum()

    positive_percentage = (
        positive / total_posts
    ) * 100

    negative_percentage = (
        negative / total_posts
    ) * 100

    neutral_percentage = (
        neutral / total_posts
    ) * 100

    overall_score = (
        sentiment_df["compound_score"]
        .mean()
    )

    aspect_scores = (
        aspect_df
        .groupby("aspect")[
            "sentiment_score"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    most_positive = (
        aspect_scores.index[0]
    )

    most_negative = (
        aspect_scores.index[-1]
    )

    most_positive_score = (
        aspect_scores.iloc[0]
    )

    most_negative_score = (
        aspect_scores.iloc[-1]
    )

    most_discussed = (
        aspect_df["aspect"]
        .value_counts()
        .index[0]
    )

    st.subheader(
        "Overall Sentiment"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Positive",
        f"{positive_percentage:.2f}%"
    )

    col2.metric(
        "Negative",
        f"{negative_percentage:.2f}%"
    )

    col3.metric(
        "Neutral",
        f"{neutral_percentage:.2f}%"
    )

    st.metric(
        "Overall Sentiment Score",
        round(overall_score, 4)
    )

    st.markdown("---")

    st.subheader(
        "Key Findings"
    )

    st.success(
        f"Most positive aspect: "
        f"{most_positive} "
        f"({most_positive_score:.4f})"
    )

    st.error(
        f"Most negative aspect: "
        f"{most_negative} "
        f"({most_negative_score:.4f})"
    )

    st.info(
        f"Most discussed aspect: "
        f"{most_discussed}"
    )

    st.markdown("---")

    st.subheader(
        "🎯 Actionable Recommendations"
    )

    if positive_percentage > negative_percentage:

        st.write(
            "✅ Positive sentiment is dominant. "
            "The organization should maintain the "
            "current customer experience."
        )

    else:

        st.write(
            "⚠️ Negative sentiment requires attention. "
            "Customer complaints should be investigated."
        )

    st.write(
        f"🔧 Improve the '{most_negative}' aspect "
        "because it has the lowest sentiment score."
    )

    st.write(
        f"⭐ Maintain the strengths associated with "
        f"'{most_positive}'."
    )

    st.write(
        f"📌 Monitor '{most_discussed}' because it is "
        "the most frequently discussed aspect."
    )


# ============================================================
# DATA & REPORTS
# ============================================================

elif page == "Data & Reports":

    st.header("📥 Data & Reports")

    st.subheader(
        "Download Complete Excel Report"
    )

    if os.path.exists(report_file):

        with open(
            report_file,
            "rb"
        ) as file:

            st.download_button(
                label="📥 Download Excel Report",
                data=file,
                file_name=(
                    "social_media_analysis_report.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )

    else:

        st.warning(
            "Excel report has not been generated yet."
        )

    st.markdown("---")

    st.subheader(
        "Analysis Files"
    )

    files = [
        "data/social_media_data.csv",
        "data/processed_social_media_data.csv",
        "data/sentiment_analysis.csv",
        "data/aspect_sentiment.csv",
        "data/topic_modeling_results.csv",
        "data/nmf_topic_results.csv",
        "data/sentiment_trend.csv",
        "reports/social_media_insights_report.txt",
        "reports/social_media_analysis_report.xlsx"
    ]

    for file in files:

        if os.path.exists(file):

            st.write(
                f"✅ {file}"
            )

        else:

            st.write(
                f"❌ {file}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Social Media Insights | "
    "NLP + Sentiment Analysis + Topic Modeling"
)