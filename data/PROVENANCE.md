# Dataset provenance — UCI Adult (Census Income)

| Field | Value |
|---|---|
| Dataset name | Adult / Census Income |
| Source URL | https://archive.ics.uci.edu/dataset/2/adult (files: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) |
| DOI | 10.24432/C5XW20 |
| Authors / publisher | Barry Becker & Ronny Kohavi, 1996; UCI Machine Learning Repository |
| License / usage terms | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Collection method | 1994 US Census (CPS) extract; donated to UCI |
| Download date (UTC) | 2026-10-23T16:32:59Z |
| Raw files | adult.data (training), adult.test (test), adult.names (schema) |
| Original size | 48,842 instances (32,561 train + 16,281 test), 14 attributes + label |
| Final size after cleaning | 45,222 instances after dropping rows with missing ( ? ) values |
| Relevant label | income: 1 if >50K else 0 |
| Quasi-identifiers (QI) | age, sex, race, marital_status, education, native_country |
| Sensitive attributes | sex (attribute-inference target), occupation (l-diversity sensitive) |
| Known limitations | Temporally stale (1994), US-centric, documented demographic skew (Ding et al. 2021); fnlwgt is a survey weight and is excluded from modelling. |
| Privacy safeguards | Public, de-identified aggregate survey microdata; no direct identifiers present. |

## File hashes (SHA-256)
```
adult.data   5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d
adult.test   a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05
adult.names  c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a
```

Processed artefacts and their hashes are written by `src/privpipe/data.py` into `data/processed/` (see `processed_manifest.json`).
