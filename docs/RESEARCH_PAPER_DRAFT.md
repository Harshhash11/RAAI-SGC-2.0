# Evaluating Markov–TF-IDF Weighted Graph-Augmented Retrieval for Domain-Specific Large Language Model Question Answering

**Manuscript status: experimental draft, NOT publication- or submission-ready.** Author names, affiliation, ethics, license attribution, citations and independently evaluated answer-quality results must be supplied by the researchers. Do not copy as a falsely completed study.

## Abstract (provisional)
A domain-specific information retrieval pipeline was implemented for question answering across AI/ML, Cloud/DevOps, Cybersecurity and Software Engineering. The system combines a TF-IDF lexical retrieval index with one-hop concept expansion over a graph whose links are scored using Markov transition probabilities and TF-IDF weights. A vector-only control and an unweighted graph ablation were implemented alongside the weighted method. The evaluation corpus consists of 145 source records, 4,771 graph-edge records and 290 unreviewed Q&A drafts. In engineering self-retrieval tests constructed from 145 document titles, vector-only retrieval achieved 88.97% top-1 self-retrieval and the unweighted and weighted graph modes each achieved 94.48%. These title-derived probes are diagnostic only and do not establish question-answering effectiveness. Independent test questions, pretrained LLM generation and fine-tuning evidence remain outstanding, so no claim of superior QA quality is made.

## 1. Introduction
Specialized IT assistants need domain-appropriate source retrieval and grounded generation. A graph may improve source discoverability by linking related technical concepts, but its value must be measured independently of text matching and model fluency. This work studies a retrieval graph applied to an existing domain corpus, rather than presenting graph-construction as the primary research result.

**Research question:** Does the Markov–TF-IDF weighted graph method improve retrieval and evidence-backed answers relative to a vector-only baseline and unweighted graph expansion, under the same corpus, retrieval depth, and generator?

## 2. Dataset
The canonical `master.csv` contains 5,206 mixed record types. There are 145 SOURCE rows, 4,771 EDGE rows, and 290 QA_DRAFT rows. Only SOURCE rows are independent documents for supervised source classification. SOURCE split sizes: 116 train, 14 validation, and 15 test. The Q&A draft labels and Wikipedia source quality have not been externally approved. Source license/attribution must be reviewed.

## 3. Methodology
TF-IDF uses word-level unigrams and bigrams. A semantic backend can optionally encode text with a pretrained sentence embedding model. Graph candidate expansion starts at query-similar SOURCE nodes, follows each SOURCE → term edge, and matches term labels to canonical source titles. Weighted mode scales graph contribution by stored Markov–TF-IDF edge weight; unweighted mode uses the same topology with unit contribution. Candidates are reranked with a fixed interpolation coefficient. A pretrained instruction-tuned model can consume retrieved evidence under a citation-constrained instruction prompt. The generator and input question must be identical for all retrieval ablations.

## 4. Experimental design
**Independent gold:** Manually write and verify domain-specific questions with evidence IDs and reference answers. Keep benchmark examples out of SFT. Report Recall@1, Recall@5, MRR, and retrieval latency across all three retrieval modes. **Generation:** Evaluate fixed-model generated answers for correctness, groundedness, completeness and citation accuracy using multiple human reviewers. Report annotator agreement and confidence intervals where possible. **Fine-tuning:** Compare the pretrained model and its genuinely reviewed-data QLoRA adapter only after training, preserving the held-out benchmark. Record hardware, software versions, hyperparameters and seeds.

## 5. Results available as of packaging
The 145 title-derived source self-retrieval diagnostic yielded vector-only 88.97%, unweighted graph 94.48%, weighted graph 94.48% top-1 self-retrieval. No independently approved benchmark or answer-quality results exist. Consequently, the hypothesis that graph weighting outperforms unweighted graph expansion remains **unproven**. An auxiliary four-domain logistic-regression source classifier achieved validation macro-F1 0.9365, holdout macro-F1 0.7667 and holdout accuracy 0.8000 on 15 documents; this is *not* LLM QA performance.

## 6. Threats to validity
Title-constructed self-retrieval questions share lexical cues with the indexed document titles, leading to an optimistic estimate. The 145-document corpus is small, specialized IT terms often have ambiguous meanings, text extraction includes formatting artifacts, and document-level source quality has not been verified. Candidate expansion can introduce semantically adjacent but irrelevant pages. Source availability and licensing affect reproducibility. Held-out human-judged questions and stronger baselines are required.

## 7. Conclusions (current evidence)
An integrated GraphRAG prototype and three controlled retrieval modes have been implemented and tested. The experiments show measurable behavior but not weighted-graph superiority or improved generative QA. The next evidence needed consists of independent gold judgments, real pretrained-model and adapter checkpoints, and evaluated source-backed generated answers.

## Required manuscript completion
Add genuine scholarly citations and related-work comparison; a properly documented independent benchmark; actual LLM inference/QLoRA logs and evaluation results; study ethics/attribution review; figures and detailed parameters; a balanced limitations discussion; and exact university formatting with original author contributions.
