import os
import json
import math
from pathlib import Path

import pandas as pd
from openai import OpenAI

INPUT_FILE = "stock_tweets_cleaned_capped.csv"
TEXT_COL = "Tweet_clean"
MODEL = "gpt-5-nano"

# test first; when it works set to None
SAMPLE_N = None

OUT_CSV = "stock_tweets_cleaned_capped_sentiment_responses_minimal.csv"
OUT_XLSX = "stock_tweets_cleaned_capped_sentiment_responses_minimal.xlsx"

LABEL_TO_SCORE = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"]
        }
    },
    "required": ["sentiment"]
}


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["sentiment", "sentiment_score", "api_error"]:
        if c not in df.columns:
            df[c] = None
    return df


def extract_output_text(resp) -> str:
    # Responses API exposes output_text directly
    text = getattr(resp, "output_text", None)
    if isinstance(text, str):
        return text.strip()
    return ""


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    path = Path(INPUT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Columns: {list(df.columns)}")

    df = ensure_cols(df)

    if SAMPLE_N is not None:
        df = df.head(SAMPLE_N).copy()

    client = OpenAI()

    for idx, row in df.iterrows():
        text = "" if pd.isna(row[TEXT_COL]) else str(row[TEXT_COL]).strip()

        try:
            resp = client.responses.create(
                model=MODEL,
                reasoning={"effort": "minimal"},
                text={
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": "tweet_sentiment",
                        "schema": SCHEMA,
                        "strict": True,
                    },
                },
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Classify the sentiment of this financial tweet."
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text
                            }
                        ],
                    },
                ],
            )

            raw = extract_output_text(resp)
            obj = json.loads(raw)

            label = obj["sentiment"]
            df.at[idx, "sentiment"] = label
            df.at[idx, "sentiment_score"] = LABEL_TO_SCORE[label]
            df.at[idx, "api_error"] = None

        except Exception as e:
            df.at[idx, "sentiment"] = None
            df.at[idx, "sentiment_score"] = None
            df.at[idx, "api_error"] = str(e)

        print(f"Processed row {idx}")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(OUT_XLSX, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()