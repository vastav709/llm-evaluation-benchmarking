import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {
    "question_id", "language", "question", "reference_answer",
    "model_name", "model_response"
}


def normalise(text: str) -> str:
    """Lowercase text and remove punctuation/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(reference: str, prediction: str):
    """Compute unigram overlap precision, recall and F1."""
    ref_tokens = normalise(reference).split()
    pred_tokens = normalise(prediction).split()

    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0

    ref_counts = {}
    pred_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = sum(min(count, pred_counts.get(token, 0))
                  for token, count in ref_counts.items())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add exact match, token-overlap and TF-IDF similarity metrics."""
    df = df.copy()

    df["exact_match"] = [
        int(normalise(ref) == normalise(pred))
        for ref, pred in zip(df["reference_answer"], df["model_response"])
    ]

    scores = [
        token_f1(ref, pred)
        for ref, pred in zip(df["reference_answer"], df["model_response"])
    ]
    df["token_precision"] = [s[0] for s in scores]
    df["token_recall"] = [s[1] for s in scores]
    df["token_f1"] = [s[2] for s in scores]

    # Fit on reference + prediction pairs. This is a lexical baseline,
    # not a factuality or deep semantic evaluation metric.
    texts = (
        df["reference_answer"].astype(str).tolist()
        + df["model_response"].astype(str).tolist()
    )
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    n = len(df)
    df["tfidf_cosine_similarity"] = [
        float(cosine_similarity(matrix[i], matrix[n + i])[0, 0])
        for i in range(n)
    ]

    return df


def build_summary(scored: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "exact_match", "token_precision", "token_recall",
        "token_f1", "tfidf_cosine_similarity"
    ]
    summary = scored.groupby("model_name")[metrics].mean().reset_index()

    if "human_rating" in scored.columns:
        summary["human_rating_mean"] = (
            pd.to_numeric(scored["human_rating"], errors="coerce")
            .groupby(scored["model_name"]).mean()
            .reindex(summary["model_name"]).values
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text-model responses against reference answers."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-dir", default="reports", help="Output folder")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required fields contain missing values. Clean the CSV first.")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "human_rating" in df.columns:
        ratings = pd.to_numeric(df["human_rating"], errors="coerce")
        invalid = ratings.notna() & ~ratings.between(1, 5)
        if invalid.any():
            raise ValueError("human_rating must be between 1 and 5.")

    scored = add_metrics(df)
    summary = build_summary(scored)

    scored.to_csv(output_dir / "evaluated_responses.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)

    low = scored.sort_values(
        ["token_f1", "tfidf_cosine_similarity"],
        ascending=[True, True]
    ).head(10)
    low.to_csv(output_dir / "low_scoring_responses.csv", index=False)

    chart_metrics = ["token_f1", "tfidf_cosine_similarity"]
    ax = summary.set_index("model_name")[chart_metrics].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0
    )
    ax.set_title("Illustrative Response Evaluation Comparison")
    ax.set_ylabel("Average score (0–1)")
    ax.set_xlabel("System label")
    ax.legend(["Token F1", "TF-IDF cosine similarity"])
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()

    print("\nEvaluation complete.")
    print(f"Rows evaluated: {len(scored)}")
    print(f"Systems compared: {scored['model_name'].nunique()}")
    print(f"Reports saved to: {output_dir.resolve()}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
