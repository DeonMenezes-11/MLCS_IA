# Which Privacy Layer Stops Which Attack?
**Security and Privacy for Machine Learning (Honors) — Individual Research Project**
Deon Ajit Menezes · Roll No. 16014223030 · Batch B1

An attack-grounded comparison of central DP, local DP, anonymisation and
compressive privacy on a single census-income benchmark, producing a
*defence-coverage matrix*: which mechanism stops which attack, at what utility cost.

## Layout
```
data/            raw + processed UCI Adult; PROVENANCE.md (sources, licence, SHA-256)
src/privpipe/    from-scratch mechanisms: data, attacks, dp, ldp, ppdm, compressive
tests/           pytest correctness checks (unbiasedness, k-anonymity, ...)
IA1/             preliminary study: scripts_ia1.py, results/, report/ (PDF)
IA2/             full study: scripts/run_all.py, make_tables_ia2.py, results/,
                 report/ (PDF), paper/ (IEEE draft PDF)
```

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m privpipe.data          # clean + hash data
PYTHONPATH=src python -m pytest tests/ -q        # mechanism correctness
python IA1/scripts_ia1.py                        # IA-1 results
python IA2/scripts/run_all.py                    # IA-2 results (multi-seed)
python IA1/make_tables_ia1.py && python IA2/make_tables_ia2.py
```
Reports compile with `pdflatex` + `bibtex` (BasicTeX; no exotic packages).

## Evidence link
Repository: https://github.com/DeonMenezes-11/MLCS_IA

## Modules covered
M1 threats/attacks · M2 differential privacy · M3 local DP · M4 privacy-preserving
data mining · M5 compressive privacy — each implemented and evaluated on the same
dataset and attack suite.
