"""IA-2 full experiment driver for Deon's project.

Runs every module's mechanism across parameter sweeps and multiple seeds,
writes one CSV per experiment into IA2/results/, and assembles the
defence-coverage matrix. Reproducible: fixed seeds throughout.

Run:  python IA2/scripts/run_all.py            # full
      QUICK=1 python IA2/scripts/run_all.py    # smaller sweeps for a smoke test
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from privpipe import data, attacks, dp, ldp, ppdm, compressive  # noqa: E402
from privpipe.models import make_model, BASELINES  # noqa: E402
from privpipe.metrics import utility, mean_std  # noqa: E402

RES = os.path.join(ROOT, "IA2", "results")
os.makedirs(RES, exist_ok=True)
RAW = os.path.join(ROOT, "data", "raw")
QUICK = os.environ.get("QUICK") == "1"

SEEDS = [0, 1, 2, 3, 4]
HEAVY_SEEDS = [0, 1, 2] if not QUICK else [0]
EPS = [0.5, 1.0, 2.0, 4.0, 8.0] if not QUICK else [1.0, 4.0]
DELTA = 1e-5


def save(df, name):
    p = os.path.join(RES, name)
    pd.DataFrame(df).to_csv(p, index=False)
    print(f"  -> wrote {name} ({len(df)} rows)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ds = data.build(RAW, seed=42)
    y_sex_tr = (ds.df_train["sex"] == "Male").astype(int).to_numpy()
    y_sex_te = (ds.df_test["sex"] == "Male").astype(int).to_numpy()
    df_all = data.load_clean(RAW)

    # =============================================================== #
    # 1. Membership inference: loss-threshold + shadow, per baseline
    # =============================================================== #
    log("1. membership inference suite")
    rows = []
    for name in BASELINES:
        lt_auc, sh_auc, lt_tpr = [], [], []
        for s in HEAVY_SEEDS:
            m = make_model(name, seed=s).fit(ds.X_train, ds.y_train)
            lt = attacks.loss_threshold_mia(m, ds.X_train, ds.y_train, ds.X_test, ds.y_test, rng=s)
            lt_auc.append(lt["auc"]); lt_tpr.append(lt["tpr_at_1pct_fpr"])
            # subsample the shadow pool so RF/MLP shadow models stay tractable
            pool = np.vstack([ds.X_train, ds.X_test])
            pool_y = np.concatenate([ds.y_train, ds.y_test])
            pr = np.random.default_rng(s).choice(len(pool), min(20000, len(pool)), replace=False)
            sh = attacks.shadow_mia(m, lambda: make_model(name, seed=s),
                                    pool[pr], pool_y[pr],
                                    ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                                    n_shadow=(3 if QUICK else 4), seed=s)
            sh_auc.append(sh["auc"])
        rows.append({"model": name,
                     "loss_mia_auc": mean_std(lt_auc)[0], "loss_mia_auc_std": mean_std(lt_auc)[1],
                     "loss_mia_tpr1": mean_std(lt_tpr)[0],
                     "shadow_mia_auc": mean_std(sh_auc)[0], "shadow_mia_auc_std": mean_std(sh_auc)[1]})
    save(rows, "mia_suite.csv")

    # =============================================================== #
    # 2. DP supervised learning: DP-NB, DP-LR, DP-SGD vs epsilon
    # =============================================================== #
    log("2. DP supervised sweep (NB / LR / SGD)")
    rows = []
    for eps in EPS:
        # DP-NB
        accs, mias = [], []
        for s in HEAVY_SEEDS:
            m = dp.DPNaiveBayes(eps=eps, n_bins=10, seed=s).fit(ds.X_train, ds.y_train)
            pr = m.predict_proba(ds.X_test)[:, 1]
            accs.append(utility(ds.y_test, (pr > .5).astype(int), pr)["f1_pos"])
            mias.append(attacks.loss_threshold_mia(m, ds.X_train, ds.y_train, ds.X_test, ds.y_test, rng=s)["auc"])
        rows.append({"method": "DP-NB", "eps": eps, "f1": mean_std(accs)[0],
                     "f1_std": mean_std(accs)[1], "mia_auc": mean_std(mias)[0]})
        # DP-LR (diffprivlib objective perturbation, bounded data norm)
        accs, mias = [], []
        for s in HEAVY_SEEDS:
            m = dp.fit_dp_logistic(ds.X_train, ds.y_train, eps=eps, seed=s)
            pr = m.predict_proba(ds.X_test)[:, 1]
            accs.append(utility(ds.y_test, (pr > .5).astype(int), pr)["f1_pos"])
            mias.append(attacks.loss_threshold_mia(m, ds.X_train, ds.y_train, ds.X_test, ds.y_test, rng=s)["auc"])
        rows.append({"method": "DP-LR", "eps": eps, "f1": mean_std(accs)[0],
                     "f1_std": mean_std(accs)[1], "mia_auc": mean_std(mias)[0]})
        # DP-SGD (opacus)
        f1s, eps_sp = [], []
        for s in HEAVY_SEEDS:
            proba, achieved = dp.train_dpsgd_mlp(
                ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                target_eps=eps, delta=DELTA, epochs=(6 if QUICK else 12),
                batch_size=1024, seed=s)
            f1s.append(utility(ds.y_test, proba.argmax(1), proba[:, 1])["f1_pos"])
            eps_sp.append(achieved)
        rows.append({"method": "DP-SGD", "eps": eps, "f1": mean_std(f1s)[0],
                     "f1_std": mean_std(f1s)[1], "mia_auc": np.nan,
                     "eps_spent": mean_std(eps_sp)[0]})
        log(f"   eps={eps} done")
    # non-private references
    for name, key in [("nb", "NB"), ("lr", "LR"), ("mlp", "MLP")]:
        f1s = []
        for s in HEAVY_SEEDS:
            m = make_model(name, seed=s).fit(ds.X_train, ds.y_train)
            pr = m.predict_proba(ds.X_test)[:, 1]
            f1s.append(utility(ds.y_test, (pr > .5).astype(int), pr)["f1_pos"])
        rows.append({"method": f"non-private {key}", "eps": np.inf,
                     "f1": mean_std(f1s)[0], "f1_std": mean_std(f1s)[1], "mia_auc": np.nan})
    save(rows, "dp_supervised.csv")

    # =============================================================== #
    # 3. DP k-means (unsupervised, 2.4)
    # =============================================================== #
    log("3. DP k-means")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    base_km = KMeans(n_clusters=5, n_init=5, random_state=0).fit(ds.X_train[:8000])
    rows = []
    for eps in EPS:
        aris = []
        for s in HEAVY_SEEDS:
            cent, _ = dp.dp_kmeans(ds.X_train[:8000], k=5, eps=eps, n_iter=8, seed=s)
            d = ((ds.X_train[:8000][:, None, :] - cent[None]) ** 2).sum(2)
            assign = d.argmin(1)
            aris.append(adjusted_rand_score(base_km.labels_, assign))
        rows.append({"eps": eps, "ari_vs_nonprivate": mean_std(aris)[0],
                     "ari_std": mean_std(aris)[1]})
    save(rows, "dp_kmeans.csv")

    # =============================================================== #
    # 4. LDP: frequency oracles + numeric oracles + LDP-NB
    # =============================================================== #
    log("4. LDP oracles + LDP-NB")
    edu = ds.df_train["education"].astype("category")
    codes = edu.cat.codes.to_numpy(); d_card = len(edu.cat.categories)
    true = np.bincount(codes, minlength=d_card) / len(codes)
    rows = []
    for eps in EPS:
        for mech in ["GRR", "OUE", "OLH"]:
            mses = []
            for s in SEEDS:
                if mech == "GRR":
                    o = ldp.GRR(d_card, eps, seed=s); est = o.estimate_freq(o.perturb(codes))
                elif mech == "OUE":
                    o = ldp.UnaryEncoding(d_card, eps, True, seed=s); est = o.estimate_freq(o.perturb(codes))
                else:
                    o = ldp.OLH(d_card, eps, seed=s); rep, sd = o.perturb(codes); est = o.estimate_freq(rep, sd)
                mses.append(float(np.mean((est - true) ** 2)))
            rows.append({"kind": "freq", "mech": mech, "eps": eps,
                         "mse": mean_std(mses)[0], "mse_std": mean_std(mses)[1]})
    # numeric mean oracles on standardised age in [-1,1]
    age = ds.X_train[:, data.NUMERIC_COLS.index("age")]
    age = np.clip(age / (np.abs(age).max() + 1e-9), -1, 1)
    true_mean = age.mean()
    for eps in EPS:
        for mech in ["Laplace", "Duchi", "Piecewise"]:
            errs = []
            for s in SEEDS:
                if mech == "Laplace":
                    out = ldp.laplace_ldp(age, eps, -1, 1, seed=s)
                elif mech == "Duchi":
                    out = ldp.duchi_mean(age, eps, seed=s)
                else:
                    out = ldp.piecewise(age, eps, seed=s)
                errs.append((out.mean() - true_mean) ** 2)
            rows.append({"kind": "mean", "mech": mech, "eps": eps,
                         "mse": mean_std(errs)[0], "mse_std": mean_std(errs)[1]})
    save(rows, "ldp_oracles.csv")

    # LDP Naive Bayes classifier accuracy vs eps
    rows = []
    for eps in EPS:
        f1s = []
        for s in HEAVY_SEEDS:
            m = ldp.LDPNaiveBayes(eps=eps, n_bins=8, seed=s).fit(ds.X_train, ds.y_train)
            pr = m.predict_proba(ds.X_test)[:, 1]
            f1s.append(utility(ds.y_test, (pr > .5).astype(int), pr)["f1_pos"])
        rows.append({"eps": eps, "f1": mean_std(f1s)[0], "f1_std": mean_std(f1s)[1]})
    save(rows, "ldp_nb.csv")

    # =============================================================== #
    # 5. Anonymisation: Mondrian across k + l-diversity/t-closeness + utility
    # =============================================================== #
    log("5. anonymisation sweep")
    from sklearn.preprocessing import OneHotEncoder
    rows = []
    for k in ([5, 20] if QUICK else [2, 5, 10, 20, 50]):
        anon = ppdm.mondrian(df_all, data.QI_COLS, k=k)
        kach = ppdm.check_k_anonymity(anon, data.QI_COLS)
        ld = ppdm.l_diversity(anon, data.SENSITIVE_DIVERSITY)
        hom = ppdm.homogeneity_attack(anon, data.SENSITIVE_DIVERSITY)
        tc = ppdm.t_closeness(anon, data.SENSITIVE_DIVERSITY)
        # utility: train RF on anonymised generalised QI (midpoint/str) + other cols
        util = _anon_utility(anon, df_all)
        rows.append({"k": k, "achieved_k": kach, "min_l": ld["min_l"],
                     "homogeneous_classes": ld["n_homogeneous_classes"],
                     "disclosed_frac": hom["frac_disclosed"], "t_closeness": tc,
                     "n_classes": ld["n_classes"], "util_acc": util})
    save(rows, "anonymisation.csv")

    # =============================================================== #
    # 6. Output modification as an MIA defence (4.5)
    # =============================================================== #
    log("6. output-modification defence")
    rf = make_model("rf", seed=0).fit(ds.X_train, ds.y_train)
    rows = []
    for mode in ["full-confidence", "rounded-1dp", "label-only"]:
        def scored(X):
            p = rf.predict_proba(X)
            if mode == "rounded-1dp":
                p = ppdm.rounded_confidence(p, 1)
            elif mode == "label-only":
                p = ppdm.label_only(p)
            return p
        m_scores = scored(ds.X_train)[np.arange(len(ds.y_train)), ds.y_train]
        n_scores = scored(ds.X_test)[np.arange(len(ds.y_test)), ds.y_test]
        from privpipe.metrics import mia_summary
        n = min(len(m_scores), len(n_scores))
        summ = mia_summary(m_scores[:n], n_scores[:n])
        rows.append({"output_mode": mode, "mia_auc": summ["auc"],
                     "mia_advantage": summ["advantage"], "mia_tpr1": summ["tpr_at_1pct_fpr"]})
    save(rows, "output_mod.csv")

    # =============================================================== #
    # 7. Compressive privacy: PCA/DCA/pa-DCA/random across p
    # =============================================================== #
    log("7. compressive privacy sweep")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rows = []
    for p in ([10, 40] if QUICK else [2, 5, 10, 20, 40]):
        for method in ["PCA", "DCA", "pa-DCA", "random"]:
            ia, sa, re = [], [], []
            for s in ([0] if method in ("PCA", "DCA", "pa-DCA") else HEAVY_SEEDS):
                if method == "PCA":
                    W = compressive.pca(ds.X_train, p)
                elif method == "DCA":
                    W = compressive.dca(ds.X_train, ds.y_train, p, rho=1.0)
                elif method == "pa-DCA":
                    W = compressive.privacy_aware_dca(ds.X_train, ds.y_train, y_sex_tr, p, 1.0, 1.0)
                else:
                    W = compressive.random_projection(ds.X_train, p, seed=s)
                Ztr, Zte = compressive.project(ds.X_train, W), compressive.project(ds.X_test, W)
                sc = StandardScaler().fit(Ztr); Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
                ia.append(LogisticRegression(max_iter=1000).fit(Ztr, ds.y_train).score(Zte, ds.y_test))
                sa.append(LogisticRegression(max_iter=1000).fit(Ztr, y_sex_tr).score(Zte, y_sex_te))
                re.append(compressive.reconstruction_error(ds.X_test, W))
            rows.append({"method": method, "p": p, "income_acc": mean_std(ia)[0],
                         "sex_acc": mean_std(sa)[0], "recon_err": mean_std(re)[0]})
    save(rows, "compressive.csv")

    # =============================================================== #
    # 8. Layered defence: DCA compression + DP-SGD
    # =============================================================== #
    log("8. layered DCA + DP-SGD")
    rows = []
    W = compressive.privacy_aware_dca(ds.X_train, ds.y_train, y_sex_tr, 20, 1.0, 1.0)
    Ztr, Zte = compressive.project(ds.X_train, W), compressive.project(ds.X_test, W)
    sc = StandardScaler().fit(Ztr); Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
    for eps in ([1.0, 4.0] if QUICK else [1.0, 2.0, 4.0, 8.0]):
        f1s = []
        for s in HEAVY_SEEDS:
            proba, _ = dp.train_dpsgd_mlp(Ztr, ds.y_train, Zte, ds.y_test,
                                          target_eps=eps, delta=DELTA,
                                          epochs=(6 if QUICK else 12), batch_size=1024, seed=s)
            f1s.append(utility(ds.y_test, proba.argmax(1), proba[:, 1])["f1_pos"])
        sens = LogisticRegression(max_iter=1000).fit(Ztr, y_sex_tr).score(Zte, y_sex_te)
        rows.append({"eps": eps, "f1": mean_std(f1s)[0], "sex_acc": sens,
                     "layers": "pa-DCA(p=20)+DP-SGD"})
    save(rows, "layered.csv")

    # =============================================================== #
    # 9. ACSIncome external-validity replication
    # =============================================================== #
    log("9. ACSIncome replication")
    try:
        _acs_replication(RES, HEAVY_SEEDS)
    except Exception as e:  # noqa
        log(f"   ACS replication skipped: {e}")
        with open(os.path.join(RES, "acs_status.txt"), "w") as f:
            f.write(f"ACSIncome replication unavailable: {e}\n")

    log("ALL IA-2 EXPERIMENTS COMPLETE")


def _anon_utility(anon, df_all):
    """Train a RandomForest on the anonymised (generalised) QI table and report
    held-out income accuracy."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    work = anon.copy()
    for c in data.QI_COLS:
        if c in data.NUMERIC_COLS:
            # generalised "lo-hi" -> midpoint
            def mid(v):
                try:
                    lo, hi = str(v).split("-")[:2]
                    return (float(lo) + float(hi)) / 2
                except Exception:
                    return float(str(v).split("-")[0])
            work[c] = work[c].map(mid)
    # one-hot the categorical (generalised string) QIs + remaining categoricals
    feat = [c for c in data.NUMERIC_COLS + data.CATEGORICAL_COLS]
    X = pd.get_dummies(work[feat].astype({c: str for c in data.CATEGORICAL_COLS}),
                       columns=[c for c in data.CATEGORICAL_COLS])
    y = work[data.TARGET].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X.to_numpy(float), y, test_size=0.3, random_state=0, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def _acs_replication(res_dir, seeds):
    """Replicate the headline comparison on ACSIncome (folktables, 2018 CA)."""
    from folktables import ACSDataSource, ACSIncome
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                        root_dir=os.path.join(ROOT, "data", "raw", "acs"))
    acs = src.get_data(states=["CA"], download=True)
    X, y, _ = ACSIncome.df_to_numpy(acs)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(30000, len(X)), replace=False)
    X, y = X[idx], y[idx].astype(int)
    n = len(X); ntr = int(n * 0.7)
    sc = StandardScaler().fit(X[:ntr])
    Xtr, Xte = sc.transform(X[:ntr]), sc.transform(X[ntr:])
    ytr, yte = y[:ntr], y[ntr:]
    rows = []
    # baseline RF + MIA
    rf = make_model("rf", seed=0).fit(Xtr, ytr)
    mia = attacks.loss_threshold_mia(rf, Xtr, ytr, Xte, yte, rng=0)
    rows.append({"setting": "baseline RF", "f1": utility(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])["f1_pos"],
                 "mia_auc": mia["auc"]})
    # DP-SGD at eps=4
    f1s = []
    for s in seeds:
        proba, _ = dp.train_dpsgd_mlp(Xtr, ytr, Xte, yte, target_eps=4.0, delta=DELTA,
                                      epochs=10, batch_size=1024, seed=s)
        f1s.append(utility(yte, proba.argmax(1), proba[:, 1])["f1_pos"])
    rows.append({"setting": "DP-SGD eps=4", "f1": mean_std(f1s)[0], "mia_auc": np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, "acs_replication.csv"), index=False)
    print("  -> wrote acs_replication.csv")


if __name__ == "__main__":
    main()
