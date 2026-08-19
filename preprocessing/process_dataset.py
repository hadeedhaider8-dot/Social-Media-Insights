import pandas as pd
from preprocessing.text_preprocessor import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    stem_tokens,
    lemmatize_tokens
)


# -----------------------------
# Load Dataset
# -----------------------------
INPUT_FILE = "data/social_media_data.csv"
OUTPUT_FILE = "data/processed_social_media_data.csv"


def process_dataset():

    # Read CSV
    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset loaded: {len(df)} posts")

    # -----------------------------
    # Clean Text
    # -----------------------------
    df["clean_text"] = df["text"].apply(clean_text)

    # -----------------------------
    # Tokenization
    # -----------------------------
    df["tokens"] = df["clean_text"].apply(tokenize_text)

    # -----------------------------
    # Stopword Removal
    # -----------------------------
    df["filtered_tokens"] = df["tokens"].apply(remove_stopwords)

    # -----------------------------
    # Stemming
    # -----------------------------
    df["stemmed_tokens"] = df["filtered_tokens"].apply(stem_tokens)

    # -----------------------------
    # Lemmatization
    # -----------------------------
    df["lemmatized_tokens"] = df["stemmed_tokens"].apply(
        lemmatize_tokens
    )

    # Convert token lists into text
    df["processed_text"] = df["lemmatized_tokens"].apply(
        lambda tokens: " ".join(tokens)
    )

    # -----------------------------
    # Save Processed Dataset
    # -----------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print("Preprocessing completed successfully!")
    print(f"Processed dataset saved to: {OUTPUT_FILE}")

    # Show preview
    print("\nProcessed Data Preview:")
    print(
        df[
            [
                "text",
                "clean_text",
                "processed_text"
            ]
        ].head()
    )


if __name__ == "__main__":
    process_dataset()