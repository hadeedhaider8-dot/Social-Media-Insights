import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# Topic Distributions
# ---------------------------------

lda_topics = {
    "Topic 1": 4,
    "Topic 2": 8,
    "Topic 3": 3
}

nmf_topics = {
    "Topic 1": 2,
    "Topic 2": 6,
    "Topic 3": 7
}


# ---------------------------------
# Create Comparison DataFrame
# ---------------------------------

comparison_df = pd.DataFrame({
    "LDA": lda_topics,
    "NMF": nmf_topics
})


print("=" * 60)
print("LDA vs NMF TOPIC COMPARISON")
print("=" * 60)

print("\nTopic Distribution Comparison:")
print(comparison_df)


# ---------------------------------
# Plot Comparison
# ---------------------------------

comparison_df.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("LDA vs NMF Topic Distribution")
plt.xlabel("Topics")
plt.ylabel("Number of Posts")

plt.xticks(rotation=0)

plt.legend(
    title="Topic Modeling Method"
)

plt.tight_layout()
plt.show()


print("\nLDA vs NMF comparison completed successfully!")