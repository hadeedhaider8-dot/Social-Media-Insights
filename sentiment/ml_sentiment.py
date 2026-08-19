import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------
# File Path
# ---------------------------------
DATA_FILE = "data/sentiment_analysis.csv"


# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("MACHINE LEARNING SENTIMENT ANALYSIS")
print("=" * 60)

print(f"\nTotal posts: {len(df)}")

print("\nOriginal Sentiment Distribution:")
print(df["sentiment"].value_counts())


# ---------------------------------
# Remove Classes with Less Than 2 Posts
# ---------------------------------
class_counts = df["sentiment"].value_counts()

valid_classes = class_counts[
    class_counts >= 2
].index

df = df[
    df["sentiment"].isin(valid_classes)
].copy()

print("\nDataset after removing very small classes:")
print(df["sentiment"].value_counts())


# ---------------------------------
# Features and Target
# ---------------------------------
X = df["processed_text"].fillna("")
y = df["sentiment"]


# ---------------------------------
# Train/Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print(f"\nTraining posts: {len(X_train)}")
print(f"Testing posts: {len(X_test)}")


# ---------------------------------
# TF-IDF
# ---------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF transformation completed.")


# ---------------------------------
# Naive Bayes
# ---------------------------------
nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(
    X_test_tfidf
)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)


# ---------------------------------
# SVM
# ---------------------------------
svm_model = LinearSVC(
    random_state=42
)

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_predictions = svm_model.predict(
    X_test_tfidf
)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)


# ---------------------------------
# Model Comparison
# ---------------------------------
print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(
    f"Naive Bayes Accuracy: {nb_accuracy:.2%}"
)

print(
    f"SVM Accuracy:         {svm_accuracy:.2%}"
)


# ---------------------------------
# Naive Bayes Report
# ---------------------------------
print("\n" + "=" * 60)
print("NAIVE BAYES CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        nb_predictions,
        zero_division=0
    )
)


# ---------------------------------
# SVM Report
# ---------------------------------
print("\n" + "=" * 60)
print("SVM CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        svm_predictions,
        zero_division=0
    )
)


print("\nMachine Learning sentiment analysis completed successfully!")