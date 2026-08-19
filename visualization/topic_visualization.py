import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# Load Topic Modeling Results
# ---------------------------------
DATA_FILE = "data/topic_modeling_results.csv"

df = pd.read_csv(DATA_FILE)


# ---------------------------------
# Topic Distribution
# ---------------------------------
topic_counts = df["topic"].value_counts().sort_index()


print("=" * 60)
print("LDA TOPIC VISUALIZATION")
print("=" * 60)

print("\nTopic Distribution:")
print(topic_counts)


# ---------------------------------
# Bar Chart
# ---------------------------------
plt.figure(figsize=(8, 5))

sns.barplot(
    x=topic_counts.index.astype(str),
    y=topic_counts.values
)

plt.title("LDA Topic Distribution")
plt.xlabel("Topic")
plt.ylabel("Number of Posts")

# Display values on bars
for i, value in enumerate(topic_counts.values):
    plt.text(
        i,
        value + 0.1,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.show()


# ---------------------------------
# Pie Chart
# ---------------------------------
plt.figure(figsize=(7, 7))

plt.pie(
    topic_counts.values,
    labels=[
        f"Topic {topic}"
        for topic in topic_counts.index
    ],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("LDA Topic Percentage Distribution")

plt.tight_layout()
plt.show()


print("\nTopic visualization completed successfully!")