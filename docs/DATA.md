# Data card: NHTSA complaints corpus (v1)

**Source:** NHTSA Office of Defects Investigation, complaints flat file
(`https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip`, data dictionary `CMPL.txt`).
Public U.S. government data. Narrative field: `CDESCR`.
Source zip SHA-256 for v1: `6683d502d0bd9dea77bdba036ce3c7516ca1d400b2597da26b54d3aac969aff7`.

**Reproduce:**
```bash
python scripts/download_nhtsa.py   # -> data/interim/complaints_raw.parquet
python scripts/prepare_data.py     # -> data/processed/v1/{train,val,test}.parquet + manifest.json
```

## Pipeline

| Step | Rows |
|---|---:|
| Flat-file rows (one per complaint *component*) | 2,261,996 |
| Empty narratives dropped | 39 |
| Duplicates dropped (exact repeats, then repeats after cleaning) | 634,949 |
| Shorter than 20 characters after cleaning, dropped | 6,378 |
| **Narratives kept** | **1,620,630** |
| train / val / test | 1,457,986 / 81,673 / 80,971 |
| characters (train) | 702M |

Splits are assigned by a hash of the cleaned text (90/5/5), so the same narrative always
lands in the same split, and dedup runs **before** splitting, so no narrative appears in
both train and test (verified: zero overlap).

## Cleaning rules (`src/forgegpt/clean.py`)

Each rule was added after profiling the raw data. Counts are unique raw narratives the
rule changed.

| Rule | Hits | Why |
|---|---:|---|
| Lowercase everything | 1.63M | Text before 2021 is ~100% ALL CAPS, 2022+ is ~1% caps. Keeping case would split the corpus into two "dialects" and waste tokenizer vocabulary on `BRAKE` vs `brake`. |
| Remove analyst tags (`*AK`, `*DT*JB`, `***NAR***`) | 739k | NHTSA filing initials, not complaint text. `*NOT*` emphasis is kept. |
| Mask phone numbers -> `[phone]` | 27k | Personal information. |
| Unify NHTSA redaction marker -> `[xxx]` | 27k | Appears as `[XXX]`, `XXX`, `XXXX`... |
| Remove FOIA legal footer (2 wordings) | 26k | Boilerplate appended by NHTSA. |
| Mask URLs -> `[url]` | 11k | Not useful to model; may identify people. |
| Mask emails -> `[email]`, 17-char VINs -> `[vin]` | 758 / 235 | Personal information. |
| Normalize characters | 61k | Smart quotes/dashes -> ASCII; older records store them with the high byte stripped (e.g. `’` as `0x19`), which is mapped back. Invisible control, format and private-use characters removed. |
| Decode HTML entities | 524 | e.g. `&#65279;`. |
| Collapse whitespace | 1.08M | Runs of spaces/tabs -> one space. |

Personal fields in the flat file (VIN, city, dealer details, operator name) are never read.

## Known limitations

- **Free-text addresses** (street, city, ZIP) inside a few older narratives are not masked.
  Reliable address detection needs an NER model; the volume is small.
- **~68k narratives are NHTSA staff summaries** of phone complaints ("the contact owns a
  2015 ford edge. the contact stated..."). They are kept: they are real domain text, but
  the model will learn this template. Worth noting when reading generated samples.
- **Near-duplicates** (e.g. form letters with small edits) are not removed; only exact
  duplicates after cleaning.
- Truncation: 0.08% of narratives hit the 2,048-character field limit and end mid-sentence.
