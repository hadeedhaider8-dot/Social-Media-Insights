import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# Model Results
# ---------------------------------
models = [
    "Naive Bayes",
    "SVM"
]

accuracies = [
    0.80,
    0.80
]


# ---------------------------------
# Print Results
# ---------------------------------
print("=" * 50)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 50)

for model, accuracy in zip(models, accuracies):
    print(f"{model}: {accuracy:.0%}")


# ---------------------------------
# Create Bar Chart
# ---------------------------------
plt.figure(figsize=(8, 5))

sns.barplot(
    x=models,
    y=accuracies
)

plt.title("Sentiment Classification Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

# Show accuracy on bars
for i, accuracy in enumerate(accuracies):
    plt.text(
        i,
        accuracy + 0.02,
        f"{accuracy:.0%}",
        ha="center"
    )

plt.tight_layout()
plt.show()


print("\nML model comparison visualization completed successfully!")