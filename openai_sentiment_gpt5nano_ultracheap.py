#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

INPUT_FILE = "stock_tweets.csv"
MODEL = "gpt-5-nano"
MAX_REQUESTS_PER_BATCH = 5000
POLL_SECONDS = 20

OUT_CSV = "stock_tweets_gpt5nano_min_sentiment.csv"
OUT_XLSX = "stock_tweets_gpt5nano_min_sentiment.xlsx"
CHECKPOINT_CSV = "stock_tweets_gpt5nano_min_checkpoint.csv"

SYSTEM_PROMPT = (
    "Classify tweet sentiment.\n"
    "Return exactly: label|score|confidence\n"
    "label: positive or neutral or negative\n"
    "score: -1 to 1\n"
    "confidence: 0 to 1\n"
    "No extra text."
)


def find_text_column(df: pd.DataFrame) -> str:
    candidates = ["tweets", "Tweet", "tweet", "text", "Text", "content", "Content"]
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for c in ["tweets", "tweet", "text", "content"]:
        if c in lowered:
            return lowered[c]
    raise ValueError(f"Could not find tweet text column. Columns are: {list(df.columns)}")


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["sentiment", "sentiment_score", "sentiment_confidence", "batch_error"]:
        if col not in df.columns:
            df[col] = None
    return df


def make_batch_jsonl(df_chunk: pd.DataFrame, text_col: str, batch_no: int) -> Path:
    path = Path(f"tweets_min_batch_{batch_no}.jsonl")
    with path.open("w", encoding="utf-8") as f:
        for idx, row in df_chunk.iterrows():
            tweet = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
            obj = {
                "custom_id": f"row-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": tweet},
                    ],
                    "max_completion_tokens": 20
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return path


def upload_batch_file(client: OpenAI, path: Path) -> str:
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    return uploaded.id


def create_batch_job(client: OpenAI, input_file_id: str) -> str:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def wait_for_batch(client: OpenAI, batch_id: str):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id}: status={batch.status}")
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            return batch
        time.sleep(POLL_SECONDS)


def download_file_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    data = content.read()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data


def parse_result_text(text: str):
    if text is None:
        return None, None, None, "empty_output"

    raw = text.strip().replace("```", "").strip()
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 3:
        return None, None, None, f"bad_format:{raw[:120]}"

    label = parts[0].lower()
    if label not in {"positive", "neutral", "negative"}:
        return None, None, None, f"bad_label:{raw[:120]}"

    try:
        score = float(parts[1])
        conf = float(parts[2])
    except Exception:
        return None, None, None, f"bad_numbers:{raw[:120]}"

    score = max(-1.0, min(1.0, score))
    conf = max(0.0, min(1.0, conf))
    return label, score, conf, None


def extract_message_content(response_obj: dict):
    try:
        return response_obj["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return None


def apply_batch_output(df: pd.DataFrame, output_text: str):
    updated = 0
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")
        m = re.match(r"row-(\d+)", custom_id)
        if not m:
            continue
        row_idx = int(m.group(1))

        if obj.get("error"):
            df.at[row_idx, "batch_error"] = str(obj["error"])
            continue

        content = extract_message_content(obj)
        label, score, conf, err = parse_result_text(content)

        df.at[row_idx, "sentiment"] = label
        df.at[row_idx, "sentiment_score"] = score
        df.at[row_idx, "sentiment_confidence"] = conf
        df.at[row_idx, "batch_error"] = err
        updated += 1
    return updated


def save_checkpoint(df: pd.DataFrame):
    df.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(input_path)
    text_col = find_text_column(df)
    df = ensure_output_columns(df)

    print(f"Loaded {len(df):,} rows. Using text column: {text_col}")

    client = OpenAI()

    pending_indices = df.index[df["sentiment"].isna()].tolist()
    print(f"Pending rows: {len(pending_indices):,}")

    if not pending_indices:
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        df.to_excel(OUT_XLSX, index=False)
        print(f"Saved: {OUT_CSV}")
        print(f"Saved: {OUT_XLSX}")
        return

    n_batches = (len(pending_indices) + MAX_REQUESTS_PER_BATCH - 1) // MAX_REQUESTS_PER_BATCH
    print(f"Sequential mini-batches to run: {n_batches}")

    for batch_no, start_pos in enumerate(range(0, len(pending_indices), MAX_REQUESTS_PER_BATCH), start=1):
        end_pos = min(start_pos + MAX_REQUESTS_PER_BATCH, len(pending_indices))
        batch_indices = pending_indices[start_pos:end_pos]
        df_chunk = df.loc[batch_indices, [text_col]]

        print(f"--- Mini-batch {batch_no}/{n_batches}: {len(df_chunk):,} requests ---")

        jsonl_path = make_batch_jsonl(df_chunk, text_col, batch_no)
        print(f"Wrote {jsonl_path.name}")

        input_file_id = upload_batch_file(client, jsonl_path)
        print(f"Uploaded batch input file: {input_file_id}")

        batch_id = create_batch_job(client, input_file_id)
        print(f"Created batch job: {batch_id}")

        batch = wait_for_batch(client, batch_id)
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} ended with status={batch.status}. errors={getattr(batch, 'errors', None)}")

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} completed but no output_file_id was returned.")

        output_text = download_file_text(client, output_file_id)
        updated = apply_batch_output(df, output_text)
        print(f"Updated rows from batch output: {updated:,}")

        save_checkpoint(df)
        print(f"Checkpoint saved: {CHECKPOINT_CSV}")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(OUT_XLSX, index=False)
    print("Done.")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
