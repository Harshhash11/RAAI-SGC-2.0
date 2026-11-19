# Part 1 methodology — reproducible dataset engineering

## Scope and unit of observation

This Part 1 dataset addresses **Technology and IT** through AI/ML, Cloud/DevOps, Cybersecurity, and Software Engineering. The core unit of provenance is a Wikipedia page revision. Rows in the one master CSV fall into three record types:

- `SOURCE`: one page-level document with title, URL, source revision, retrieval timestamp and original extract.
- `EDGE`: a directed concept-to-phrase association extracted from the source; both surface phrase and exact source evidence span are retained.
- `QA_DRAFT`: an extractive question-answer template, not expert-verified conversational data.

The master CSV is **not** a conventional single-table classification dataset; fields not applicable to a given `record_type` intentionally remain blank. The data dictionary describes all 32 columns. For an LLM, the **eligible training observations are reviewed Q&A records**, not source and edge rows merely counted as examples.

## Collection and lineage

The supplied baseline was collected with the bundled Ubuntu MediaWiki collection and spaCy graph-construction pipeline and was later corrected offline. The Wikipedia page title, URL, revision, and retrieval timestamp are retained. Original V3.4 source data and the first corrected baseline are kept outside the final output file. The `input/BASELINE_149_SOURCES.csv` file is the exact immutable input for this consolidation. No live API collection or article-specific source checking is claimed in this finalization step.

Four out-of-scope article titles were excluded from the *analysis candidate*: `Ancient language`, `Arabic`, `Neural network (biology)`, and `Software Engineering Notes`. Their IDs, URLs and reasons are kept in `output/EXCLUDED_SOURCES.csv`. All associated edges and Q&A rows were removed together. Other borderline or specialized sources remain **review candidates**, not automated approvals.

## Mathematical formulation

For a given source concept A and extracted target B, the source-level transition probability is computed from candidate term occurrence frequencies. Edge importance follows the previous RAAI-SGC formulation:

`W(A,B) = P(A→B) × (1 + TF-IDF(B))`

A normalization check requires that outgoing stored probabilities per source sum to 1, within numerical tolerance. Every edge's stored weight is recomputed for the arithmetic check. The weights do **not** demonstrate factual correctness or causal relationships. Association graph edges do **not** imply ontology relation types such as `is-a` or `causes`.

## Evidence and licensing

The stored `target_surface` and evidence character offsets allow source-excerpt verification. Draft answer text must be fully contained within a matching evidence quote, which must be contained within the source text. These automated checks only demonstrate **textual grounding** and **internal consistency**. They cannot verify truth, authority, conceptual relevance or copyright clearance. Wikipedia has reuse/attribution obligations; exact permissions and suitability for the planned model use must be reviewed with attribution requirements recorded. `license_status=REVIEW_CC_BY_SA` is not an approval.

## Split methodology and leakage risks

Source-grouped train/validation/test labels are retained. All rows derived from the same source have the same split, avoiding direct source-ID leakage. Related articles and near-duplicate excerpts may nevertheless transmit overlapping facts across splits. An **independent gold benchmark is not included or claimed**. The included template must be populated and reviewed separately; evaluation questions from it must not enter fine-tuning data.

## Research limitations

This is a 145-source candidate corpus with limited breadth and many unverified extractive QA drafts. Longer technical/operational and cross-domain answers are not supported simply by adding more templates. The project should report downstream model comparisons only after using a genuinely reviewed QA set and held-out benchmark. Any reported model metrics must come from actual execution. Source, term, and Q&A review queues identify the remaining verification work.
