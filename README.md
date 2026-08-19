#  Social Media Insights

A Python-based Social Media Analytics system that performs sentiment analysis, topic modeling, aspect-based sentiment analysis, trend analysis, and business insight generation.

##  Project Overview

This project analyzes social media posts to understand customer opinions, discover important topics, identify product and service aspects, and generate actionable business insights.

The system combines Natural Language Processing (NLP), Machine Learning, Topic Modeling, Data Visualization, and Streamlit to provide an interactive analytics dashboard.

## ✨ Features

- Text preprocessing
- Exploratory Data Analysis
- VADER sentiment analysis
- Naive Bayes sentiment classification
- SVM sentiment classification
- LDA topic modeling
- NMF topic modeling
- LDA vs NMF comparison
- Aspect-based sentiment analysis
- Sentiment trend analysis
- Business insight generation
- Excel report generation
- Interactive Streamlit dashboard

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Gensim
- Matplotlib
- Seaborn
- WordCloud
- VADER Sentiment
- Streamlit
- OpenPyXL

## 📁 Project Structure

```text
Social-Media-Insights/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── social_media_data.csv
│   ├── processed_social_media_data.csv
│   ├── sentiment_analysis.csv
│   ├── aspect_sentiment.csv
│   ├── topic_modeling_results.csv
│   ├── nmf_topic_results.csv
│   └── sentiment_trend.csv
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_preprocessor.py
│   └── process_dataset.py
│
├── sentiment/
│   ├── vader_sentiment.py
│   ├── ml_sentiment.py
│   └── aspect_sentiment.py
│
├── topic_modeling/
│   ├── lda_topics.py
│   └── nmf_topics.py
│
├── visualization/
│   ├── eda.py
│   ├── sentiment_visualization.py
│   ├── ml_comparison.py
│   ├── topic_visualization.py
│   ├── topic_comparison.py
│   └── aspect_visualization.py
│
├── utils/
│   └── trend_analysis.py
│
└── reports/
    ├── insight_generator.py
    ├── export_report.py
    ├── social_media_insights_report.txt
    └── social_media_analysis_report.xlsx
