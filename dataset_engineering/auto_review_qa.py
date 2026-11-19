import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/review/QA_REVIEW_QUEUE.csv"
OUTPUT = ROOT / "data/review/QA_AUTO_REVIEW.csv"

def tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def assess(row):
    question = (row.get("question") or "").strip()
    answer = (row.get("answer_draft") or "").strip()
    evidence = (row.get("evidence_quote") or "").strip()

    flags = []

    if not question or not answer:
        flags.append("MISSING_QUESTION_OR_ANSWER")

    if len(tokens(answer)) < 8:
        flags.append("SHORT_ANSWER")

    if not evidence:
        flags.append("MISSING_EVIDENCE")

    if evidence and answer:
        overlap = len(tokens(answer) & tokens(evidence)) / max(1, len(tokens(answer)))
        if overlap < 0.5:
            flags.append("LOW_EVIDENCE_OVERLAP")

    if "what does the source explain" in question.lower():
        flags.append("TEMPLATE_CONTEXT_QUESTION")

    if question and answer and not (tokens(question) & tokens(answer)):
        flags.append("LOW_QUESTION_ANSWER_OVERLAP")

    row["auto_review_status"] = (
        "NEEDS_REVIEW" if flags else "CANDIDATE_FOR_REVIEW"
    )
    row["auto_review_flags"] = ";".join(flags)
    return row

with INPUT.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = [assess(row) for row in reader]

output_columns = list(dict.fromkeys(
    columns + ["auto_review_status", "auto_review_flags"]
))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=output_columns)
    writer.writeheader()
    writer.writerows(rows)

print("\n=== RAAI-SGC 2.0 AUTOMATED QA REVIEW ===")
print("Total questions:", len(rows))
print("Results:", dict(Counter(r["auto_review_status"] for r in rows)))
print("Flags:", dict(Counter(
    flag
    for row in rows
    for flag in row["auto_review_flags"].split(";")
    if flag
)))
print("Saved:", OUTPUT)
print("NOTE: No questions have been human-approved automatically.")
