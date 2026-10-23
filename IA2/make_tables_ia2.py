"""Generate all IA-2 LaTeX tables, figures and scalar macros from the result
CSVs. No number in the IA-2 report or paper is typed by hand."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "report", "tables")
FIG = os.path.join(HERE, "report", "figs")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
PAL = ["#1f4e79", "#c0504d", "#2e7d32", "#d9a441", "#6a4c93", "#7f7f7f"]
plt.rcParams.update({"figure.figsize": (5, 3.1), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def w(name, text):
    open(os.path.join(TAB, name), "w").write(text)


def f3(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def tbl(colspec, header, rows, name):
    w(name, "\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n"
      % (colspec, header, "\n".join(rows)))


R = lambda f: pd.read_csv(os.path.join(RES, f))
macros = {}

# ---- 1. MIA suite ----------------------------------------------------------
m = R("mia_suite.csv")
names = {"lr": "Logistic Reg.", "nb": "Naive Bayes", "rf": "Random Forest", "mlp": "MLP"}
rows = [f"{names.get(r['model'], r['model'])} & {f3(r['loss_mia_auc'])} & "
        f"{f3(r['loss_mia_tpr1'])} & {f3(r['shadow_mia_auc'])} \\\\" for _, r in m.iterrows()]
tbl("lrrr", "Model & Loss-MIA AUC & TPR@1\\%FPR & Shadow-MIA AUC", rows, "tab_mia.tex")
rf = m[m.model == "rf"].iloc[0]
macros["RFlossMIA"] = f3(rf["loss_mia_auc"]); macros["RFshadowMIA"] = f3(rf["shadow_mia_auc"])

# ---- 2. DP supervised ------------------------------------------------------
d = R("dp_supervised.csv")
fin = d[np.isfinite(d["eps"])]
# table: eps vs F1 for each method
piv = fin.pivot_table(index="eps", columns="method", values="f1")
rows = []
for eps, r in piv.iterrows():
    rows.append(f"{eps:.1f} & " + " & ".join(f3(r.get(mth, np.nan)) for mth in ["DP-NB", "DP-LR", "DP-SGD"]) + " \\\\")
np_ref = d[~np.isfinite(d["eps"])]
tbl("lrrr", "$\\varepsilon$ & DP-NB $F_1$ & DP-LR $F_1$ & DP-SGD $F_1$", rows, "tab_dpsup.tex")
# figure: F1 vs eps
fig, ax = plt.subplots()
for i, mth in enumerate(["DP-NB", "DP-LR", "DP-SGD"]):
    s = fin[fin.method == mth].sort_values("eps")
    if len(s):
        ax.plot(s["eps"], s["f1"], "o-", color=PAL[i], label=mth)
for mth, key in [("non-private MLP", "MLP")]:
    ref = np_ref[np_ref.method == mth]
    if len(ref):
        ax.axhline(ref.iloc[0]["f1"], ls="--", color=PAL[5], label="non-private MLP")
ax.set_xlabel("privacy budget $\\varepsilon$"); ax.set_ylabel("$F_1$ (>\\$50K)")
ax.legend(fontsize=7); fig.savefig(os.path.join(FIG, "dp_supervised.pdf")); plt.close(fig)

# ---- 3. DP k-means ---------------------------------------------------------
try:
    k = R("dp_kmeans.csv")
    rows = [f"{r['eps']:.1f} & {f3(r['ari_vs_nonprivate'])} \\\\" for _, r in k.iterrows()]
    tbl("lr", "$\\varepsilon$ & ARI vs non-private", rows, "tab_dpkmeans.tex")
except Exception:
    pass

# ---- 4. LDP oracles --------------------------------------------------------
L = R("ldp_oracles.csv")
freq = L[L.kind == "freq"]; num = L[L.kind == "mean"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for i, mech in enumerate(sorted(freq.mech.unique())):
    s = freq[freq.mech == mech].sort_values("eps")
    axes[0].plot(s["eps"], s["mse"], "o-", color=PAL[i], label=mech)
axes[0].set_yscale("log"); axes[0].set_title("Frequency oracles"); axes[0].legend(fontsize=7)
axes[0].set_xlabel("$\\varepsilon$"); axes[0].set_ylabel("MSE")
for i, mech in enumerate(sorted(num.mech.unique())):
    s = num[num.mech == mech].sort_values("eps")
    axes[1].plot(s["eps"], s["mse"], "o-", color=PAL[i], label=mech)
axes[1].set_yscale("log"); axes[1].set_title("Numeric mean oracles"); axes[1].legend(fontsize=7)
axes[1].set_xlabel("$\\varepsilon$"); axes[1].set_ylabel("MSE")
fig.savefig(os.path.join(FIG, "ldp_oracles.pdf")); plt.close(fig)

# LDP-NB
ln = R("ldp_nb.csv")
rows = [f"{r['eps']:.1f} & {f3(r['f1'])} \\\\" for _, r in ln.iterrows()]
tbl("lr", "$\\varepsilon_{local}$ & LDP-NB $F_1$", rows, "tab_ldpnb.tex")

# ---- 5. Anonymisation ------------------------------------------------------
a = R("anonymisation.csv")
rows = [f"{int(r['k'])} & {int(r['achieved_k'])} & {int(r['min_l'])} & "
        f"{int(r['homogeneous_classes'])} & {f3(r['disclosed_frac'])} & "
        f"{f3(r['t_closeness'])} & {f3(r['util_acc'])} \\\\" for _, r in a.iterrows()]
tbl("lrrrrrr", "Target $k$ & Ach.\\ $k$ & Min $\\ell$ & Homog.\\ & Disclosed & $t$ & Util.\\ acc", rows, "tab_anon2.tex")
fig, ax = plt.subplots()
ax.plot(a["k"], a["util_acc"], "o-", color=PAL[0], label="utility (income acc.)")
ax2 = ax.twinx()
ax2.plot(a["k"], a["disclosed_frac"], "s--", color=PAL[1], label="homogeneity disclosure")
ax.set_xlabel("$k$"); ax.set_ylabel("income accuracy"); ax2.set_ylabel("frac.\\ disclosed")
ax.set_xscale("log")
fig.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.9, 0.9))
fig.savefig(os.path.join(FIG, "anon.pdf")); plt.close(fig)

# ---- 6. Output modification ------------------------------------------------
o = R("output_mod.csv")
rows = [f"{r['output_mode']} & {f3(r['mia_auc'])} & {f3(r['mia_advantage'])} & {f3(r['mia_tpr1'])} \\\\"
        for _, r in o.iterrows()]
tbl("lrrr", "Released output & MIA AUC & MIA adv. & TPR@1\\%", rows, "tab_outmod.tex")
full = o[o.output_mode == "full-confidence"].iloc[0]
lab = o[o.output_mode == "label-only"].iloc[0]
macros["MIAfull"] = f3(full["mia_auc"]); macros["MIAlabel"] = f3(lab["mia_auc"])

# ---- 7. Compressive --------------------------------------------------------
c = R("compressive.csv")
fig, ax = plt.subplots()
for i, meth in enumerate(["PCA", "DCA", "pa-DCA", "random"]):
    s = c[c.method == meth].sort_values("p")
    ax.plot(s["income_acc"], s["sex_acc"], "o-", color=PAL[i], label=meth)
ax.set_xlabel("income accuracy (utility)"); ax.set_ylabel("sex accuracy (leakage)")
ax.legend(fontsize=7); fig.savefig(os.path.join(FIG, "compressive.pdf")); plt.close(fig)
# table at p=20
c20 = c[c.p == c.p.max()] if "p" in c else c
rows = [f"{r['method']} & {int(r['p'])} & {f3(r['income_acc'])} & {f3(r['sex_acc'])} & {f3(r['recon_err'])} \\\\"
        for _, r in c[c.p == sorted(c.p.unique())[len(c.p.unique())//2]].iterrows()]
tbl("lrrrr", "Projection & $p$ & Income acc. & Sex acc. & Recon.\\ err.", rows, "tab_comp2.tex")

# ---- 8. Layered ------------------------------------------------------------
try:
    ly = R("layered.csv")
    rows = [f"{r['eps']:.1f} & {f3(r['f1'])} & {f3(r['sex_acc'])} \\\\" for _, r in ly.iterrows()]
    tbl("lrr", "$\\varepsilon$ & $F_1$ (utility) & sex acc.\\ (leakage)", rows, "tab_layered.tex")
except Exception:
    pass

# ---- 9. ACS replication ----------------------------------------------------
try:
    ac = R("acs_replication.csv")
    rows = [f"{r['setting']} & {f3(r['f1'])} & {f3(r['mia_auc']) if np.isfinite(r['mia_auc']) else '--'} \\\\"
            for _, r in ac.iterrows()]
    tbl("lrr", "Setting (ACSIncome, CA 2018) & $F_1$ & MIA AUC", rows, "tab_acs.tex")
    macros["ACSok"] = "yes"
except Exception:
    macros["ACSok"] = "no"

# ---- 10. Coverage matrix ---------------------------------------------------
# Qualitative synthesis assembled from the quantitative results above.
cov = [
    ("Membership inference", "DP-SGD / DP-NB", "\\cmark", "Central DP bounds per-record influence"),
    ("Membership inference", "Output modification", "\\cmark", "Removes the confidence signal"),
    ("Attribute inference (sex)", "Privacy-aware DCA", "\\cmark", "Suppresses the sensitive discriminant"),
    ("Linkage / re-identification", "$k$-anonymity (Mondrian)", "\\cmark", "Generalises quasi-identifiers"),
    ("Homogeneity disclosure", "$\\ell$-diversity / $t$-closeness", "\\cmark", "Diversifies each class"),
    ("Collection-time disclosure", "Local DP", "\\cmark", "No trusted curator needed"),
    ("Membership inference", "$k$-anonymity", "\\xmark", "Protects the table, not the model"),
    ("Attribute inference", "Central DP alone", "\\pmark", "Partial; not targeted at the sensitive attr."),
]
rows = [f"{a_} & {b} & {c_} & {d} \\\\" for a_, b, c_, d in cov]
tbl("p{3.3cm}p{3.0cm}cp{4.6cm}", "Attack & Mechanism & Stops? & Why", rows, "tab_coverage.tex")

# ---- macros ----------------------------------------------------------------
with open(os.path.join(TAB, "macros_ia2.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("IA-2 tables/figures/macros written.")
