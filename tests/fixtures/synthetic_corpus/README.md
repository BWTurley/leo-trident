# Synthetic ASME-Style Corpus

All content in this directory is **original and fictional**. It was written
specifically for the Leo Trident eval framework and does not reproduce any
real ASME Boiler & Pressure Vessel Code text.

## Purpose

Offline evaluation of the retrieval pipeline (BM25, dense search, PPR,
reranker, and relevance judge) before real corpus ingestion.

## Structure

- Paragraph IDs (e.g., UG-22, UW-11, QW-200) mimic the ASME BPVC
  numbering convention but contain synthetic content.
- Cross-references between paragraphs use citation phrases recognized
  by the ASME parser's REFERENCE_TYPE_PATTERNS.
- `_metadata.json` contains ground-truth topics and cross-references
  extracted by running the parser over each paragraph.

## Coverage

| Prefix | Part                             | Count |
|--------|----------------------------------|-------|
| UG-*   | General Requirements (VIII-1)    |  35   |
| UW-*   | Welded Vessels (VIII-1)          |  19   |
| UCS-*  | Carbon and Low Alloy Steel       |  12   |
| UHA-*  | High Alloy Steel                 |   6   |
| QW-*   | Welding Qualifications (IX)      |  18   |

## Citation type distribution

The corpus targets roughly 40% mandatory, 25% conditional, 25%
informational, and 10% unclassified cross-reference types.
