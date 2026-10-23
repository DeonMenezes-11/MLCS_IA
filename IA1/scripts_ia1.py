"""IA-1 preliminary experiments.

Deliberately preliminary: a single utility baseline suite, a first attack suite,
and ONE mechanism per module at fixed settings. Everything is written to
IA1/results/*.csv and IA1/report/figs/*.pdf so the report cites real numbers.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from privpipe import data, attacks, dp, ldp, ppdm, compressive, plots  # noqa: E402
from privpipe.models import make_model, BASELINES  # noqa: E402
from privpipe.metrics import utility, mean_std  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(ROOT, "IA1", "report", "figs")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
RAW = os.path.join(ROOT, "data", "raw")
plots.style()
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = [0, 1, 2]


def main():
    ds = data.build(RAW, seed=42)
    print(f"train={len(ds.y_train)} test={len(ds.y_test)} feats={ds.n_features}")

    # ------------------------------------------------------------------ #
    # 1. Baselines + loss-threshold MIA
    # ------------------------------------------------------------------ #
    rows = []
    for name in BASELINES:
        accs, f1s, aucs, mia_aucs, mia_advs = [], [], [], [], []
        for s in SEEDS:
            m = make_model(name, seed=s)
            m.fit(ds.X_train, ds.y_train)
            proba = m.predict_proba(ds.X_test)[:, 1]
            u = utility(ds.y_test, (proba > 0.5).astype(int), proba)
            accs.append(u["accuracy"]); f1s.append(u["f1_pos"]); aucs.append(u.get("roc_auc", np.nan))
            mia = attacks.loss_threshold_mia(m, ds.X_train, ds.y_train, ds.X_test, ds.y_test, rng=s)
            mia_aucs.append(mia["auc"]); mia_advs.append(mia["advantage"])
        rows.append({
            "model": name,
            "accuracy": mean_std(accs)[0], "accuracy_std": mean_std(accs)[1],
            "f1_pos": mean_std(f1s)[0],
            "roc_auc": mean_std(aucs)[0],
            "mia_auc": mean_std(mia_aucs)[0], "mia_auc_std": mean_std(mia_aucs)[1],
            "mia_advantage": mean_std(mia_advs)[0],
        })
        print(f"  {name}: acc={rows[-1]['accuracy']:.3f} mia_auc={rows[-1]['mia_auc']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "baselines.csv"), index=False)

    # ------------------------------------------------------------------ #
    # 2. Attribute inference ("dropping sex isn't enough")
    # ------------------------------------------------------------------ #
    feat_cols = [c for c in (data.NUMERIC_COLS + data.CATEGORICAL_COLS) if c != "sex"]
    ai = attacks.attribute_inference(ds.df_train, ds.df_test, feat_cols, "sex", seed=0)
    pd.DataFrame([ai]).to_csv(os.path.join(RESULTS, "attribute_inference.csv"), index=False)
    print(f"  attribute inference (sex): acc={ai['attack_accuracy']:.3f} base={ai['majority_baseline']:.3f}")

    # ------------------------------------------------------------------ #
    # 3. QI uniqueness / linkage risk
    # ------------------------------------------------------------------ #
    qi_rows = []
    df_all = data.load_clean(RAW)
    for ab in [1, 5, 10]:
        q = attacks.qi_uniqueness(df_all, data.QI_COLS, age_bin=ab)
        q["age_bin"] = ab
        qi_rows.append(q)
    pd.DataFrame(qi_rows).to_csv(os.path.join(RESULTS, "qi_uniqueness.csv"), index=False)
    print(f"  QI uniqueness (age_bin=1): frac_unique={qi_rows[0]['frac_unique']:.3f}")

    # ------------------------------------------------------------------ #
    # 4. M2: DP Naive Bayes eps sweep (+ non-private reference)
    # ------------------------------------------------------------------ #
    dpnb_rows = []
    Xtr_num = ds.X_train; Xte_num = ds.X_test
    for eps in [0.5, 1.0, 2.0, 5.0]:
        m = dp.DPNaiveBayes(eps=eps, n_bins=10, seed=0).fit(Xtr_num, ds.y_train)
        pr = m.predict_proba(Xte_num)[:, 1]
        u = utility(ds.y_test, (pr > 0.5).astype(int), pr)
        dpnb_rows.append({"eps": eps, **u})
    # non-private GaussianNB reference
    gnb = make_model("nb").fit(Xtr_num, ds.y_train)
    prg = gnb.predict_proba(Xte_num)[:, 1]
    ug = utility(ds.y_test, (prg > 0.5).astype(int), prg); ug["eps"] = np.inf
    dpnb_rows.append(ug)
    pd.DataFrame(dpnb_rows).to_csv(os.path.join(RESULTS, "dpnb_sweep.csv"), index=False)

    fig, ax = plt.subplots()
    d = pd.DataFrame(dpnb_rows)
    fin = d[np.isfinite(d["eps"])]
    ax.plot(fin["eps"], fin["accuracy"], "o-", color=plots.PALETTE[0], label="DP-NB accuracy")
    ax.axhline(ug["accuracy"], ls="--", color=plots.PALETTE[1], label="non-private NB")
    ax.set_xlabel("privacy budget $\\varepsilon$"); ax.set_ylabel("test accuracy")
    ax.legend(); plots.save(fig, os.path.join(FIGS, "dpnb_sweep.pdf"))

    # ------------------------------------------------------------------ #
    # 5. M3: LDP frequency oracle MSE vs eps on 'education'
    # ------------------------------------------------------------------ #
    edu = ds.df_train["education"].astype("category")
    codes = edu.cat.codes.to_numpy()
    d_card = len(edu.cat.categories)
    true = np.bincount(codes, minlength=d_card) / len(codes)
    ldp_rows = []
    for eps in [0.5, 1.0, 2.0, 4.0]:
        for mech in ["GRR", "OUE"]:
            if mech == "GRR":
                o = ldp.GRR(d_card, eps, seed=0); est = o.estimate_freq(o.perturb(codes))
            else:
                o = ldp.UnaryEncoding(d_card, eps, optimized=True, seed=0)
                est = o.estimate_freq(o.perturb(codes))
            mse = float(np.mean((est - true) ** 2))
            ldp_rows.append({"mech": mech, "eps": eps, "mse": mse})
    pd.DataFrame(ldp_rows).to_csv(os.path.join(RESULTS, "ldp_freq.csv"), index=False)

    fig, ax = plt.subplots()
    L = pd.DataFrame(ldp_rows)
    for i, mech in enumerate(["GRR", "OUE"]):
        sub = L[L.mech == mech]
        ax.plot(sub["eps"], sub["mse"], "o-", color=plots.PALETTE[i], label=mech)
    ax.set_yscale("log"); ax.set_xlabel("privacy budget $\\varepsilon$")
    ax.set_ylabel("frequency MSE"); ax.legend()
    plots.save(fig, os.path.join(FIGS, "ldp_mse.pdf"))

    # ------------------------------------------------------------------ #
    # 6. M4: Mondrian k-anonymity utility + l-diversity (preliminary, k=5)
    # ------------------------------------------------------------------ #
    anon = ppdm.mondrian(df_all, data.QI_COLS, k=5)
    k_ach = ppdm.check_k_anonymity(anon, data.QI_COLS)
    ld = ppdm.l_diversity(anon, data.SENSITIVE_DIVERSITY)
    hom = ppdm.homogeneity_attack(anon, data.SENSITIVE_DIVERSITY)
    tc = ppdm.t_closeness(anon, data.SENSITIVE_DIVERSITY)
    anon_row = {"k": 5, "achieved_k": k_ach, "min_l": ld["min_l"],
                "homogeneous_classes": ld["n_homogeneous_classes"],
                "disclosed_frac": hom["frac_disclosed"], "t_closeness": tc,
                "n_classes": ld["n_classes"]}
    pd.DataFrame([anon_row]).to_csv(os.path.join(RESULTS, "anonymization.csv"), index=False)
    print(f"  Mondrian k=5: achieved_k={k_ach} min_l={ld['min_l']} disclosed={hom['frac_disclosed']:.3f}")

    # ------------------------------------------------------------------ #
    # 7. M5: preliminary PCA vs DCA at a single target dimension
    # ------------------------------------------------------------------ #
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    y_sex = (ds.df_train["sex"] == "Male").astype(int).to_numpy()
    y_sex_te = (ds.df_test["sex"] == "Male").astype(int).to_numpy()
    comp_rows = []
    p = 10
    for method in ["PCA", "DCA", "pa-DCA", "random"]:
        if method == "PCA":
            W = compressive.pca(ds.X_train, p)
        elif method == "DCA":
            W = compressive.dca(ds.X_train, ds.y_train, p, rho=1.0)
        elif method == "pa-DCA":
            W = compressive.privacy_aware_dca(ds.X_train, ds.y_train, y_sex, p, rho=1.0, alpha=1.0)
        else:
            W = compressive.random_projection(ds.X_train, p, seed=0)
        Ztr = compressive.project(ds.X_train, W); Zte = compressive.project(ds.X_test, W)
        sc = StandardScaler().fit(Ztr)
        Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
        util_clf = LogisticRegression(max_iter=1000).fit(Ztr, ds.y_train)
        util_acc = util_clf.score(Zte, ds.y_test)
        sens_clf = LogisticRegression(max_iter=1000).fit(Ztr, y_sex)
        sens_acc = sens_clf.score(Zte, y_sex_te)
        recon = compressive.reconstruction_error(ds.X_test, W)
        comp_rows.append({"method": method, "p": p, "income_acc": util_acc,
                          "sex_acc": sens_acc, "recon_err": recon})
    pd.DataFrame(comp_rows).to_csv(os.path.join(RESULTS, "compressive_prelim.csv"), index=False)
    print("  compressive:", [(r["method"], round(r["income_acc"], 3), round(r["sex_acc"], 3)) for r in comp_rows])

    # summary
    summary = {
        "n_train": int(len(ds.y_train)), "n_test": int(len(ds.y_test)),
        "n_features": int(ds.n_features),
        "positive_rate": float(ds.y_test.mean()),
    }
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("IA-1 experiments complete.")


if __name__ == "__main__":
    main()
