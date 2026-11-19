import csv
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data/review"
SUGGESTIONS = REVIEW / "QA_SUGGESTIONS_TO_VERIFY.csv"
QUEUE = REVIEW / "QA_REVIEW_QUEUE.csv"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

fields, qa_rows = read_csv(QUEUE)
_, suggestions = read_csv(SUGGESTIONS)

approved = [
    r for r in suggestions
    if r.get("human_review_status", "").strip().upper() == "APPROVED"
    and r.get("generation_status") == "SUGGESTED"
]

qa_by_id = {r["question_id"]: r for r in qa_rows}

print(f"\nApproved suggestions found: {len(approved)}")
print("The original question will be preserved.")
print("Source-document approvals will NOT be changed.\n")

reviewer = input("Enter your reviewer name: ").strip()
if not reviewer:
    raise SystemExit("Reviewer name is required. No changes made.")

accepted = 0
skipped = 0

for index, s in enumerate(approved, 1):
    q = qa_by_id.get(s["question_id"])

    if not q:
        print("Skipping unmatched ID:", s["question_id"])
        skipped += 1
        continue

    if q["source_id"] != s["source_id"] or q["split"] != "train":
        print("Skipping source/split mismatch:", s["question_id"])
        skipped += 1
        continue

    answer = s.get("suggested_answer", "").strip()
    if not answer:
        print("Skipping empty answer:", s["question_id"])
        skipped += 1
        continue

    print(f"\n{'=' * 65}")
    print(f"RECORD {index}/{len(approved)}")
    print("ID:", s["question_id"])
    print("Source:", s["source_title"])
    print("\nORIGINAL QUESTION:", q["question"])
    print("\nSUGGESTED QUESTION:", s["suggested_question"])
    print("\nSUGGESTED ANSWER:", answer)
    print("\nEVIDENCE:", q["evidence_quote"])

    print("\nApprove only if the answer:")
    print(" - correctly answers the ORIGINAL question")
    print(" - is factually accurate")
    print(" - is relevant and sufficiently complete")
    print(" - is supported by the displayed evidence")

    decision = input("Confirm all four checks? [y/N/q]: ").strip().lower()

    if decision == "q":
        break

    if decision != "y":
        skipped += 1
        continue

    q["approved_answer"] = answer

    for key in (
        "answer_factuality",
        "answer_relevance",
        "answer_completeness",
        "evidence_adequacy",
    ):
        q[key] = "APPROVED"

    q["reviewer"] = reviewer
    q["review_date"] = date.today().isoformat()
    q["review_notes"] = (
        "Human-reviewed Qwen suggestion; checked against "
        "original question and source evidence."
    )

    accepted += 1

if accepted:
    backup = QUEUE.with_name(
        f"QA_REVIEW_QUEUE.before_transfer_{date.today().isoformat()}.csv"
    )

    if backup.exists():
        raise SystemExit(
            f"Backup already exists: {backup}. "
            "Rename it before running this transfer again."
        )

    shutil.copy2(QUEUE, backup)

    with QUEUE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(qa_rows)

    print("\nBackup saved:", backup)

print("\n=== TRANSFER SUMMARY ===")
print("Newly confirmed Q&A records:", accepted)
print("Skipped or not confirmed:", skipped)
print("Official queue:", QUEUE)
print("Source reviews were not modified.")
