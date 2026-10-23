"""Generate LaTeX table fragments and scalar macros from the IA-1 result CSVs.

The report \\input's these, so no number in the report is typed by hand.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "report", "tables")
os.makedirs(TAB, exist_ok=True)


def w(name, text):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def f3(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


# ---- baselines -------------------------------------------------------------
b = pd.read_csv(os.path.join(RES, "baselines.csv"))
rows = []
names = {"lr": "Logistic Regression", "nb": "Gaussian Naive Bayes",
         "rf": "Random Forest", "mlp": "MLP"}
for _, r in b.iterrows():
    rows.append(f"{names.get(r['model'], r['model'])} & {f3(r['accuracy'])} & "
                f"{f3(r['f1_pos'])} & {f3(r['roc_auc'])} & {f3(r['mia_auc'])} & "
                f"{f3(r['mia_advantage'])} \\\\")
w("tab_baselines.tex",
  "\\begin{tabular}{lrrrrr}\n\\toprule\nModel & Acc. & $F_1$ & ROC-AUC & "
  "MIA AUC & MIA adv. \\\\\n\\midrule\n" + "\n".join(rows) +
  "\n\\bottomrule\n\\end{tabular}\n")

# ---- DP-NB sweep -----------------------------------------------------------
d = pd.read_csv(os.path.join(RES, "dpnb_sweep.csv"))
rows = []
for _, r in d.iterrows():
    eps = "$\\infty$" if not np.isfinite(r["eps"]) else f"{r['eps']:.1f}"
    rows.append(f"{eps} & {f3(r['accuracy'])} & {f3(r['balanced_accuracy'])} & "
                f"{f3(r['f1_pos'])} & {f3(r['roc_auc'])} \\\\")
w("tab_dpnb.tex",
  "\\begin{tabular}{lrrrr}\n\\toprule\n$\\varepsilon$ & Acc. & Bal. Acc. & "
  "$F_1$ & ROC-AUC \\\\\n\\midrule\n" + "\n".join(rows) +
  "\n\\bottomrule\n\\end{tabular}\n")

# ---- LDP freq MSE ----------------------------------------------------------
L = pd.read_csv(os.path.join(RES, "ldp_freq.csv"))
piv = L.pivot(index="eps", columns="mech", values="mse")
rows = []
for eps, r in piv.iterrows():
    rows.append(f"{eps:.1f} & {r['GRR']:.2e} & {r['OUE']:.2e} \\\\")
w("tab_ldp.tex",
  "\\begin{tabular}{lrr}\n\\toprule\n$\\varepsilon$ & GRR MSE & OUE MSE \\\\\n"
  "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- QI uniqueness ---------------------------------------------------------
q = pd.read_csv(os.path.join(RES, "qi_uniqueness.csv"))
rows = []
for _, r in q.iterrows():
    rows.append(f"{int(r['age_bin'])} & {f3(r['frac_unique'])} & "
                f"{f3(r['frac_le_k2'])} & {int(r['achieved_k'])} & "
                f"{int(r['n_classes'])} \\\\")
w("tab_qi.tex",
  "\\begin{tabular}{lrrrr}\n\\toprule\nAge bin & Frac. unique & Frac.\\ $\\le2$ "
  "& Achieved $k$ & \\#classes \\\\\n\\midrule\n" + "\n".join(rows) +
  "\n\\bottomrule\n\\end{tabular}\n")

# ---- anonymization ---------------------------------------------------------
a = pd.read_csv(os.path.join(RES, "anonymization.csv")).iloc[0]
w("tab_anon.tex",
  "\\begin{tabular}{lr}\n\\toprule\nProperty & Value \\\\\n\\midrule\n"
  f"Target $k$ & {int(a['k'])} \\\\\n"
  f"Achieved $k$ (min class size) & {int(a['achieved_k'])} \\\\\n"
  f"\\#equivalence classes & {int(a['n_classes'])} \\\\\n"
  f"Min $\\ell$ (distinct-$\\ell$ diversity) & {int(a['min_l'])} \\\\\n"
  f"Homogeneous classes & {int(a['homogeneous_classes'])} \\\\\n"
  f"Records disclosed by homogeneity & {f3(a['disclosed_frac'])} \\\\\n"
  f"Worst-class $t$ (total-variation) & {f3(a['t_closeness'])} \\\\\n"
  "\\bottomrule\n\\end{tabular}\n")

# ---- compressive -----------------------------------------------------------
c = pd.read_csv(os.path.join(RES, "compressive_prelim.csv"))
rows = []
for _, r in c.iterrows():
    rows.append(f"{r['method']} & {f3(r['income_acc'])} & {f3(r['sex_acc'])} & "
                f"{f3(r['recon_err'])} \\\\")
w("tab_compressive.tex",
  "\\begin{tabular}{lrrr}\n\\toprule\nProjection ($p{=}10$) & Income acc.\\ "
  "(utility) & Sex acc.\\ (leakage) & Recon.\\ err. \\\\\n\\midrule\n" +
  "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- scalar macros ---------------------------------------------------------
ai = pd.read_csv(os.path.join(RES, "attribute_inference.csv")).iloc[0]
summ = json.load(open(os.path.join(RES, "summary.json")))
best = b.loc[b["accuracy"].idxmax()]
rf = b[b["model"] == "rf"].iloc[0]
macros = {
    "NTrain": f"{summ['n_train']:,}",
    "NTest": f"{summ['n_test']:,}",
    "NFeat": str(summ["n_features"]),
    "PosRate": f"{summ['positive_rate']*100:.1f}\\%",
    "BestAcc": f3(best["accuracy"]), "BestModel": names.get(best["model"]),
    "RFMIA": f3(rf["mia_auc"]), "RFAdv": f3(rf["mia_advantage"]),
    "AIacc": f3(ai["attack_accuracy"]), "AIbase": f3(ai["majority_baseline"]),
    "AIleak": f3(ai["leakage"]),
    "FracUnique": f"{q.iloc[0]['frac_unique']*100:.1f}\\%",
    "PADCAincome": f3(c[c.method == "pa-DCA"].iloc[0]["income_acc"]),
    "PADCAsex": f3(c[c.method == "pa-DCA"].iloc[0]["sex_acc"]),
    "PCAsex": f3(c[c.method == "PCA"].iloc[0]["sex_acc"]),
}
with open(os.path.join(TAB, "macros.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote tables + macros to", TAB)
