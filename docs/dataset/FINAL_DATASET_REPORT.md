# RAAI-SGC 2.0 — final Part 1 dataset engineering report

**Status:** CONDITIONAL PASS for structural/algorithmic quality; **NOT VERIFIED** for expert fact review, licensing clearance, training eligibility and independent benchmark readiness.

## Scope and audited result

A single 32-column master CSV contains **5,206** records: **145 source pages**, **4,771** weighted extracted concept edges, and **290** extractive draft Q&A records. The prior baseline contained 149 sources; four off-scope pages and all their dependent records were quarantined, not overwritten, with source-specific reasons preserved.

| Domain | Sources |
|---|---:|
| AI / ML | 40 |
| Cloud / DevOps | 32 |
| Cybersecurity | 36 |
| Software engineering | 37 |
| **Total** | **145** |

Source-grouped data splits: **116 train**, **14 validation**, **15 test** sources. All records for one source share that source's split; this is not proof of absence of semantic/near-duplicate leakage.

## Automated test outcomes

The latest code checks unique IDs, source hash consistency, provenance fields, valid linkages, exact source evidence offsets, exact answer-to-evidence substring containment, non-self-referential edges, Markov transition normalization, and the original weighted formula. **Zero structural/algorithmic critical failures** in the consolidated dataset.

The graph uses `W(A,B) = P(A→B) × (1 + TF-IDF(B))`. This only establishes numerical consistency; no expert proof of edge relevance or model-quality improvement is implied.

## Remaining risk register

| Priority | Issue | Evidence / status | Release action |
|---|---|---|---|
| CRITICAL | Source licensing, attribution, factual verification | 145 reviews pending | Manually review all sources; record approval |
| CRITICAL | Q&A truthfulness and answer quality | 290 draft QA, none approved | Validate or rewrite with evidence |
| HIGH | Independent gold evaluation | Empty benchmark template | Construct & audit independent questions |
| HIGH | Generic graph terms | 19 distinct generic target terms auto-flagged | Expert review and controlled graph rebuild |
| MEDIUM | Short source extracts | 7 of 145 auto-flagged | Check original source or recollect |
| MEDIUM | Dataset size and question diversity | 145 sources, 2 templates/source | Record scope limitation; expand only if project requires |

## Methods and integrity

The dataset retains provider, article URL, revision ID, retrieval timestamp, source text and SHA-256 integrity hash. Graph edges preserve exact supporting source excerpts and source text offsets. Question-answer drafts retain direct evidence quotes. Human review queues are provided separately to prevent automated heuristics from being confused with scientific verification.

## Final statement for academic documentation

> We prepared a source-traceable, four-domain IT knowledge corpus with 145 Wikipedia source records, 4,771 Markov–TF-IDF-weighted semantic associations and 290 source-extractive Q&A drafts. All master CSV rows passed automated structural, formula and evidence-integrity checks. Factual and licensing review of sources and Q&A, as well as an independent benchmark, remain pending; therefore downstream training and evaluation performance are not claimed.
