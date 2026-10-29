"""Numbers and tables for the report, generated from results/ (no hand-copied values).

    python -m experiments.tables      -> report/generated/numbers.tex, tab_*.tex
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "generated"

macros: dict[str, str] = {}


def m(name: str, value) -> None:
    macros[name] = str(value)


def f3(x) -> str:
    return f"{x:.3f}"


def f2(x) -> str:
    return f"{x:.2f}"


def thin(n) -> str:
    return f"{int(n):,}".replace(",", "\\,")


def pct(x) -> str:
    return f"{100 * x:.0f}\\%"


def sel(d, **kw):
    for k, v in kw.items():
        d = d[d[k].astype(str) == str(v)]
    return d


def one(d, col="recall", **kw):
    r = sel(d, **kw)
    if r.empty:
        return float("nan")
    return float(r[col].iloc[0])


# ------------------------------------------------------------------ static benchmark

def static() -> None:
    p = RES / "static_glove_k10.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    nmax = int(d.n.max())
    g = sel(d, profile="paper", n=nmax)
    ours, ours_ms = one(g, method="folded-ip", param=64), one(g, "ms", method="folded-ip", param=64)
    ex_ms = one(g, "ms", method="exact")
    m("GlN", thin(nmax))
    m("GlOursRecallOneM", f3(ours))
    m("GlOursMsOneM", f2(ours_ms))
    m("GlExactMsOneM", f"{ex_ms:.1f}")
    m("GlSpeedupOneM", f"{ex_ms / ours_ms:.0f}")
    m("GlRerankRecallOneM", f3(one(g, method="rerank", param=100)))
    m("GlRerankKRecallOneM", f3(one(g, method="rerank", param=1000)))
    m("GlRelonlyRecallOneM", f3(one(g, method="relonly", param=64)))
    m("GlRerankRecallTenK", f3(one(sel(d, profile="paper", n=int(d.n.min())), method="rerank", param=100)))
    m("GlRerankRecallHundredK", f3(one(sel(d, profile="paper", n=100000), method="rerank", param=100)))
    kt = RES / "exact_knn_timing.json"
    if kt.exists():
        mv = json.loads(kt.read_text())[str(nmax)]["median_ms"]
        m("GlExactMvMsOneM", f"{mv:.0f}")
        m("GlSpeedupMv", f"{mv / ours_ms:.0f}")
    ab = sel(d, profile="paper", n=100000)
    if not ab.empty:
        m("AbOldAnchor", f3(sel(ab, method="folded-oldanchor").recall.max()))
        m("AbSQuarter", f3(one(ab, method="folded-s=0.25", param=128)))
        m("AbSFour", f3(one(ab, method="folded-s=4", param=128)))
        m("AbSOne", f3(one(ab, method="folded-ip", param=128)))
    t = pd.read_csv(RES / "static_town_k10.csv") if (RES / "static_town_k10.csv").exists() else None
    if t is not None:
        tp = sel(t, profile="paper")
        m("TwN", thin(t.n.iloc[0]))
        m("TwOursRecall", f3(one(tp, method="folded-ip", param=64)))
        m("TwRerankRecall", f3(one(tp, method="rerank", param=100)))
        m("TwRelonlyRecall", f3(one(tp, method="relonly", param=64)))
        m("TwExactMs", f2(one(tp, "ms", method="exact")))
        m("TwOursMs", f2(one(tp, "ms", method="folded-ip", param=64)))
        m("TwOursExactNorm", f3(one(tp, method="folded-ip-exactnorm", param=64)))
        m("TwIvfRecall", f3(one(tp, method="folded-ivf", param=16)))
        m("TwIvfMs", f2(one(tp, "ms", method="folded-ivf", param=16)))
        m("TwOldAnchor", f3(sel(tp, method="folded-oldanchor").recall.max()))
        m("TwLtwoDeep", f3(one(tp, method="folded-l2", param=512)))
        tc = sel(t, profile="code")
        m("TwIvfCodeRecall", f3(one(tc, method="folded-ivf", param=16)))
        m("TwOursCodeRecall", f3(one(tc, method="folded-ip", param=64)))
        m("TwIpDeep", f3(one(tp, method="folded-ip", param=512)))
    m("GlOursExactNorm", f3(one(g, method="folded-ip-exactnorm", param=64)))
    m("GlOursRecallDeep", f3(one(g, method="folded-ip", param=256)))
    m("GlCodeRecall", f3(one(sel(d, profile="code", n=nmax), method="folded-ip", param=64)))
    m("GlCodeRecallDeep", f3(one(sel(d, profile="code", n=nmax), method="folded-ip", param=256)))
    if not ab.empty:
        m("AbIvfRecall", f3(one(ab, method="folded-ivf", param=64)))
        m("AbIvfMs", f2(one(ab, "ms", method="folded-ivf", param=64)))
        m("AbOursMs", f2(one(ab, "ms", method="folded-ip", param=128)))
    # depth of the two range searches (exp_normef.py, a separate build of the same indexes)
    ne = RES / "static_normef.csv"
    if ne.exists():
        e = pd.read_csv(ne)
        # LaTeX macro names cannot hold digits: 32 -> A, 64 -> B, 128 -> C, 256 -> D
        depths = {"32": "A", "64": "B", "128": "C", "256": "D", "exact": "Exact"}
        for ds, tag in (("glove", "Gl"), ("town", "Tw")):
            for prof, ptag in (("paper", ""), ("code", "Code")):
                x = sel(e, dataset=ds, profile=prof, param=64)
                for depth, dtag in depths.items():
                    r = sel(x, norm_ef=depth)
                    if r.empty:
                        continue
                    m(f"Ne{tag}{ptag}{dtag}", f3(float(r.recall.iloc[0])))
                    if depth != "exact":
                        m(f"Ne{tag}{ptag}Ms{dtag}", f2(float(r.ms.iloc[0])))
    # build-to-build variation on the town memories (exp_buildvar.py)
    # together with the builds behind tab_static (static_town_k10.csv) and tab_normef (static_normef.csv)
    bv = RES / "static_town_builds.csv"
    if bv.exists():
        b = pd.read_csv(bv)
        n_builds = int(b.build.nunique())
        for prof, ptag in (("paper", "Paper"), ("code", "Code"), ("raw", "Raw")):
            x = list(sel(b, profile=prof, param=64).recall)
            if t is not None:
                x.append(one(sel(t, profile=prof), method="folded-ip", param=64))
            if ne.exists():
                y = sel(pd.read_csv(ne), dataset="town", profile=prof, param=64, norm_ef=32)
                x += list(y.recall)
            x = [v for v in x if v == v]
            m(f"Bv{ptag}Min", f3(min(x)))
            m(f"Bv{ptag}Max", f3(max(x)))
        m("BvBuilds", str(n_builds + (t is not None) + ne.exists()))
    # table: rows = methods, columns = profiles on GloVe 1M (+ ms for the paper profile) and on the town
    rows = [("Exact scan", "exact", "-"), ("Relevance only", "relonly", 64),
            ("Overfetch + re-rank, $K=100$", "rerank", 100), ("Overfetch + re-rank, $K=1000$", "rerank", 1000),
            ("Folded HNSW, $ef=64$", "folded-ip", 64), ("Folded HNSW, $ef=256$", "folded-ip", 256)]
    profs = ["paper", "code", "raw"]
    lines = [r"\begin{tabular}{l ccc r ccc}", r"\toprule",
             r" & \multicolumn{4}{c}{GloVe, " + thin(nmax) + r" memories} & \multicolumn{3}{c}{Willowbrook, "
             + (thin(t.n.iloc[0]) if t is not None else "--") + r" memories}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r"Method & paper & code & raw & ms & paper & code & raw\\", r"\midrule"]
    for lab, meth, par in rows:
        cells = []
        for pr in profs:
            cells.append(f3(one(sel(d, n=nmax, profile=pr), method=meth, param=par)))
        msv = one(sel(d, n=nmax, profile="paper"), "ms", method=meth, param=par)
        cells.append(f"{msv:.2f}" if msv < 10 else f"{msv:.1f}")
        for pr in profs:
            cells.append(f3(one(sel(t, profile=pr), method=meth, param=par)) if t is not None else "--")
        lines.append(lab + " & " + " & ".join(cells).replace("nan", "--") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_static.tex").write_text("\n".join(lines) + "\n")


def relsweep() -> None:
    p = RES / "static_relsweep.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    cols = [("relonly", 64, "rel.\\ only"), ("rerank", 100, "$K{=}100$"), ("rerank", 1000, "$K{=}1000$"),
            ("folded-s=1", 64, "$s{=}1$, 64"), ("folded-s=1", 256, "$s{=}1$, 256"),
            ("folded-s=0.5", 64, "$s{=}0.5$, 64"), ("folded-s=0.5", 256, "$s{=}0.5$, 256")]
    lines = [r"\begin{tabular}{r ccc cccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{overfetch + re-rank} & \multicolumn{4}{c}{folded HNSW ($s$, $ef$)}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
             r"relevance weight & " + " & ".join(c[2] for c in cols) + r"\\", r"\midrule"]
    for w in sorted(d.rel_weight.unique()):
        dw = d[d.rel_weight == w]
        cells = [f3(one(dw, method=mm, param=pp)) for mm, pp, _ in cols]
        lines.append(f"{w:g} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_relsweep.tex").write_text("\n".join(lines) + "\n")
    m("RsHalfMin", f3(sel(d, method="folded-s=0.5", param=256).recall.min()))
    low = d[d.rel_weight <= 2]
    m("RsLowFoldedMin", f3(sel(low, method="folded-s=1", param=64).recall.min()))
    m("RsLowRerankMax", f3(sel(low, method="rerank", param=1000).recall.max()))
    for w, tag in ((8, "Eight"), (16, "Sixteen")):
        dw = d[d.rel_weight == w]
        m(f"Rs{tag}RerankK", f3(one(dw, method="rerank", param=1000)))
        m(f"Rs{tag}Rerank", f3(one(dw, method="rerank", param=100)))
        m(f"Rs{tag}Folded", f3(one(dw, method="folded-s=1", param=64)))
        m(f"Rs{tag}FoldedDeep", f3(one(dw, method="folded-s=1", param=256)))


def normef_table() -> None:
    """Recall and query time against the depth of the two range searches (exp_normef.py)."""
    p = RES / "static_normef.csv"
    if not p.exists():
        return
    e = pd.read_csv(p)
    e = e[(e.profile == "paper") & (e.param.astype(str) == "64")]
    sizes = {ds: int(e[e.dataset == ds].n.iloc[0]) for ds in ("glove", "town")}
    lines = [r"\begin{tabular}{l rr rr}", r"\toprule",
             r" & \multicolumn{2}{c}{GloVe, " + thin(sizes["glove"]) + r"} & \multicolumn{2}{c}{Willowbrook, "
             + thin(sizes["town"]) + r"}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"range search & recall@10 & ms & recall@10 & ms\\", r"\midrule"]
    for depth in ("32", "64", "128", "256", "exact"):
        cells = []
        for ds in ("glove", "town"):
            r = e[(e.dataset == ds) & (e.norm_ef.astype(str) == depth)]
            cells.append(f3(float(r.recall.iloc[0])) if not r.empty else "--")
            ms = float(r.ms.iloc[0]) if not r.empty else float("nan")
            cells.append(f2(ms) if ms == ms else "--")
        lab = r"exact ranges ($O(n)$ scan)" if depth == "exact" else f"beam {depth}" + (" (default)" if depth == "32" else "")
        lines.append(lab + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_normef.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ HNSW study

def hnsw() -> None:
    p = RES / "hnsw_validate.csv"
    if p.exists():
        d = pd.read_csv(p)
        m("HvN", thin(d.n.iloc[0]))
        m("HvMaxDiff", f3((d.recall_ours - d.recall_hnswlib).abs().max()))
        m("HvQpsRatio", f"{np.median(d.qps_hnswlib / d.qps_ours):.1f}")
        keep = d[d.ef.isin([16, 32, 64, 128, 256, 512])]
        lines = [r"\begin{tabular}{r cc r rr}", r"\toprule",
                 r" & \multicolumn{2}{c}{recall@10} & dist.\ eval. & \multicolumn{2}{c}{queries / s}\\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
                 r"$ef$ & ours & hnswlib & ours & ours & hnswlib\\", r"\midrule"]
        for _, r in keep.iterrows():
            lines.append(f"{int(r.ef)} & {r.recall_ours:.4f} & {r.recall_hnswlib:.4f} & {r.dist_evals_ours:.0f} & "
                         f"{thin(r.qps_ours)} & {thin(r.qps_hnswlib)}" + r"\\")
        lines += [r"\midrule", f"build time & \\multicolumn{{2}}{{c}}{{}} & & {d.build_s_ours.iloc[0]:.0f} s & "
                  f"{d.build_s_hnswlib.iloc[0]:.0f} s" + r"\\", r"\bottomrule", r"\end{tabular}"]
        (OUT / "tab_validate.tex").write_text("\n".join(lines) + "\n")
    p = RES / "hnsw_fig3_d4.csv"
    if p.exists():
        d = pd.read_csv(p).groupby(["heuristic", "mL"]).cost_at_target.first().reset_index()
        h = d[d.heuristic == 1]
        m("HThreeFlat", f"{float(h[h.mL == 0].cost_at_target.iloc[0]):.0f}")
        m("HThreeBest", f"{h.cost_at_target.min():.0f}")
        m("HThreeBestML", f"{float(h.loc[h.cost_at_target.idxmin(), 'mL']):.1f}")
        ml0 = 1 / math.log(6)
        m("HThreeDefault", f"{float(h.iloc[(h.mL - ml0).abs().argsort()[:1]].cost_at_target.iloc[0]):.0f}")
        m("HThreeRatio", f"{float(h[h.mL == 0].cost_at_target.iloc[0]) / h.cost_at_target.min():.1f}")
        m("HThreeN", thin(pd.read_csv(p).n.iloc[0]))
    p = RES / "smoke" / "hnsw_fig3_d4.csv"   # the 50 000-point trial run of the same study
    if p.exists():
        raw = pd.read_csv(p)
        d = raw.groupby(["heuristic", "mL"]).cost_at_target.first().reset_index()
        h = d[d.heuristic == 1]
        m("HThreeSmallRatio", f"{float(h[h.mL == 0].cost_at_target.iloc[0]) / h.cost_at_target.min():.1f}")
        m("HThreeSmallN", thin(raw.n.iloc[0]))
    p = RES / "hnsw_fig4_d1024.csv"
    if p.exists():
        d = pd.read_csv(p).groupby(["heuristic", "mL"]).cost_at_target.first()
        m("HFourMin", f"{d.min():.0f}")
        m("HFourMax", f"{d.max():.0f}")
    p = RES / "hnsw_fig7_d10.csv"
    if p.exists():
        d = pd.read_csv(p)
        c = d[d.dataset == "100 clusters"]
        m("HSevenSimpleMax", f3(c[c.heuristic == 0].recall.max()))
        m("HSevenHeurMax", f3(c[c.heuristic == 1].recall.max()))
    p = RES / "hnsw_glove.csv"
    if p.exists():
        d = pd.read_csv(p).groupby(["heuristic", "mL"]).cost_at_target.first().reset_index()
        h = d[d.heuristic == 1]
        m("HGloveFlat", f"{float(h[h.mL == 0].cost_at_target.iloc[0]):.0f}")
        m("HGloveHier", f"{float(h[h.mL > 0].cost_at_target.iloc[0]):.0f}")
        s = d[d.heuristic == 0]
        extra = [float(s[s.mL == 0].cost_at_target.iloc[0]) / float(h[h.mL == 0].cost_at_target.iloc[0]),
                 float(s[s.mL > 0].cost_at_target.iloc[0]) / float(h[h.mL > 0].cost_at_target.iloc[0])]
        m("HGloveSimpleExtraLo", f"{100 * (min(extra) - 1):.0f}\\%")
        m("HGloveSimpleExtraHi", f"{100 * (max(extra) - 1):.0f}\\%")


# ------------------------------------------------------------------ LangChain, reference code

def langchain_and_reference() -> None:
    p = RES / "langchain.json"
    if p.exists():
        j = json.loads(p.read_text())
        m("LcRecall", f3(j["recall_langchain"]))
        m("LcDecay", f"{j['decay_rate']:g}")
        m("LcOursRecall", f3(j["recall_ours"]))
        m("LcMinN", thin(min(a["memories"] for a in j["agents"])))
        m("LcMaxN", thin(max(a["memories"] for a in j["agents"])))
    p = RES / "ga_reference_check.json"
    if p.exists():
        j = json.loads(p.read_text())["town_summary"]
        m("GaOverlap", pct(j["overlap_reference_paper"]))
        m("GaAgeRef", f"{j['age_reference']:.1f}")
        m("GaAgePaper", f"{j['age_paper']:.1f}")
        m("GaAgeCodeW", f"{j['age_code_weights']:.1f}")
        m("GaOverlapCodeW", pct(j["overlap_reference_codeweights"]))


# ------------------------------------------------------------------ town

def town_holm(d: pd.DataFrame) -> dict:
    """Paired Wilcoxon tests of every method against the exact scan (same seeds), Holm-corrected
    over all the comparisons in the town table. Returns {(burn, method, outcome): adjusted p}."""
    tests = []
    for b in sorted(d.burn.unique()):
        db = d[d.burn == b]
        for key in ("folded", "rerank100", "relonly"):
            for col in ("aware_at_party", "recalled_at_plan", "attended"):
                piv = db.pivot_table(index="seed", columns="backend_key", values=col)
                if key not in piv or "exact" not in piv:
                    continue
                pv = piv[[key, "exact"]].dropna()
                diff = pv[key] - pv["exact"]
                p = stats.wilcoxon(pv[key], pv["exact"]).pvalue if (diff != 0).any() else 1.0
                tests.append(((b, key, col), p))
    m_ = len(tests)
    adj, running = {}, 0.0
    for rank, (key, p) in enumerate(sorted(tests, key=lambda x: x[1])):
        running = max(running, min(1.0, (m_ - rank) * p))
        adj[key] = running
    return adj


def town() -> None:
    c = RES / "calibration.csv"
    if c.exists():
        cal = pd.read_csv(c)
        row = cal[(cal.talk_rate == 0.06) & (cal.share_prob == 0.5)].iloc[0]
        m("CalKnew", f"{row.aware_mean:.1f}")
        m("CalAttended", f"{row.attended_mean:.1f}")
    p = RES / "town.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.profile == "paper"]
    order = [("exact", "Exact scan (paper)"), ("folded", "Folded HNSW (ours)"),
             ("rerank100", "Overfetch + re-rank, $K{=}100$"), ("relonly", "Relevance only")]
    burns = sorted(d.burn.unique())
    lines = [r"\begin{tabular}{l l r r r r r}", r"\toprule",
             r"Run & Retrieval & memories & recall & knew & recalled & attended\\", r"\midrule"]
    holm = town_holm(d)
    for b in burns:
        db = d[d.burn == b]
        for i, (key, lab) in enumerate(order):
            dk = db[db.backend_key == key].set_index("seed")
            if dk.empty:
                continue
            cells = []
            for col in ("aware_at_party", "recalled_at_plan", "attended"):
                n = len(dk)
                mean, ci = dk[col].mean(), stats.t.ppf(0.975, n - 1) * dk[col].std(ddof=1) / np.sqrt(n)
                pv = holm.get((b, key, col), 1.0)
                mark = "$^{**}$" if pv < 0.01 else ("$^{*}$" if pv < 0.05 else "")
                cells.append(f"{mean:.1f} $\\pm$ {ci:.1f}{mark}")
            age = f"{b + 2} days" if i == 0 else ""
            lines.append(f"{age} & {lab} & {dk.mean_memories.mean():.0f} & {dk.recall.mean():.3f} & "
                         + " & ".join(cells) + r"\\")
        if b != burns[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_town.tex").write_text("\n".join(lines) + "\n")
    # paired statistics used in the text
    for b in burns:
        db = d[d.burn == b]
        piv = {c: db.pivot_table(index="seed", columns="backend_key", values=c) for c in
               ("aware_at_party", "recalled_at_plan", "attended", "recall")}
        tag = {0: "Zero", 7: "Seven", 30: "Thirty", 90: "Ninety"}.get(b)
        if tag is None:
            continue
        for key, short in (("exact", "Ex"), ("folded", "Fo"), ("rerank100", "Re"), ("relonly", "Rel")):
            if key not in piv["attended"]:
                continue
            for col, cs in (("aware_at_party", "Aware"), ("recalled_at_plan", "Recalled"), ("attended", "Att")):
                m(f"T{tag}{short}{cs}", f"{piv[col][key].mean():.1f}")
            m(f"T{tag}{short}Recall", f3(piv["recall"][key].mean()))
        if "rerank100_norecent" in piv["attended"] and "exact" in piv["attended"]:
            for col, cs in (("aware_at_party", "Aware"), ("recalled_at_plan", "Recalled"), ("attended", "Att")):
                pv = piv[col][["exact", "rerank100_norecent"]].dropna()
                m(f"T{tag}Nr{cs}", f"{pv['rerank100_norecent'].mean():.1f}")
                diff = pv["rerank100_norecent"] - pv["exact"]
                pval = stats.wilcoxon(pv["rerank100_norecent"], pv["exact"]).pvalue if (diff != 0).any() else 1.0
                m(f"T{tag}Nr{cs}P", f"{pval:.2g}")
            m(f"T{tag}NrRecall", f3(piv["recall"]["rerank100_norecent"].mean()))
        if "folded" in piv["attended"] and "exact" in piv["attended"]:
            outcomes = ("aware_at_party", "recalled_at_plan", "attended")
            same = np.logical_and.reduce([piv[c]["folded"] == piv[c]["exact"] for c in outcomes])
            maxdiff = max(float((piv[c]["folded"] - piv[c]["exact"]).abs().max()) for c in outcomes)
            m(f"T{tag}FoSame", f"{int(same.sum())}")
            m(f"T{tag}FoMaxDiff", f"{maxdiff:.0f}")
            m(f"T{tag}Seeds", f"{int(len(same))}")
    holm = town_holm(d)
    for b_, tag in ((7, "Seven"), (30, "Thirty"), (90, "Ninety")):
        for col, cs in (("aware_at_party", "Aware"), ("recalled_at_plan", "Recalled"), ("attended", "Att")):
            m(f"T{tag}Re{cs}Holm", f"{holm.get((b_, 'rerank100', col), 1.0):.2g}")
    rj = RES / "town_recall.jsonl"
    if rj.exists():
        recs = [json.loads(line) for line in rj.read_text().splitlines() if line.strip()]
        for key, short in (("rerank100", "Re"), ("relonly", "Rel"), ("folded", "Fo")):
            rows_ = [r["recall_per_day"] for r in recs
                     if r["burn"] == 90 and r["profile"] == "paper" and r["backend_key"] == key]
            if rows_:
                a_ = np.array([[np.nan if v is None else v for v in x] for x in rows_], dtype=float)
                m(f"TNinety{short}RecallEnd", f"{np.nanmean(a_[:, -3:]):.2f}")
    d90 = d[d.burn == 90].groupby("backend_key").search_ms.mean()
    m("TwSimExactMs", f2(d90["exact"]))
    m("TwSimFoldedMs", f2(d90["folded"]))
    m("TwSimMemories", thin(round(d[(d.burn == 90) & (d.backend_key == "exact")].mean_memories.mean())))
    all_runs = pd.read_csv(p)
    n_cal = int(pd.read_csv(c).seeds.sum()) if c.exists() else 0
    p_curve = RES / "town_curve.csv"
    n_curve = len(pd.read_csv(p_curve)) if p_curve.exists() else 0
    m("TownRuns", thin(len(all_runs) + n_cal + n_curve))
    # relevance-only memory against the exact scan, in towns older than the paper's two days
    drops = {"aware_at_party": [], "attended": []}
    for b in burns:
        if b == 0:
            continue
        db = d[d.burn == b]
        for col in drops:
            ex = db[db.backend_key == "exact"][col].mean()
            ro = db[db.backend_key == "relonly"][col].mean()
            drops[col].append(1 - ro / ex)
    for col, cs in (("aware_at_party", "Aware"), ("attended", "Att")):
        if drops[col]:
            m(f"TRelDrop{cs}Lo", pct(min(drops[col])))
            m(f"TRelDrop{cs}Hi", pct(max(drops[col])))


# ------------------------------------------------------------------ forgetting curves

def pc2(x) -> str:
    return f"{100 * x:.2f}\\%"


def curves() -> None:
    p = RES / "curve_fit.csv"
    if p.exists():
        d = pd.read_csv(p)
        names = (("actr", "ACT-R"), ("hyperbolic", "hyperbolic"), ("fademem", "FadeMem"))
        lines = [r"\begin{tabular}{r rr rr rr}", r"\toprule",
                 " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in names) + r"\\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                 r"$R$ & of range & relative & of range & relative & of range & relative\\", r"\midrule"]
        for R in (4, 6, 8, 10, 12):
            cells = []
            for key, _ in names:
                r = d[(d.curve == key) & (d.R == R)].iloc[0]
                cells += [pc2(r.max_range_err), pc2(r.max_rel_err)]
            lines.append(f"{R} & " + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "tab_curvefit.tex").write_text("\n".join(lines) + "\n")
        a8 = d[(d.curve == "actr") & (d.R == 8)].iloc[0]
        m("CfActrRngEight", pc2(a8.max_range_err))
        m("CfActrRelEight", pc2(a8.max_rel_err))
        m("CfFastRate", f"{a8.fastest_rate:.1f}")
        for key, tag in (("hyperbolic", "Hyp"), ("fademem", "Fade")):
            m(f"Cf{tag}RngEight", pc2(d[(d.curve == key) & (d.R == 8)].iloc[0].max_range_err))
    p = RES / "curve_fronts.json"
    if p.exists():
        j = json.loads(p.read_text())
        f = j.get("100000")
        if f:
            m("CuSegN", thin(f["segment"]))
            m("CuFrontMax", str(f["front_max"]))
            m("CuFrontMin", str(f["front_min"]))
        f = j.get("1000000")
        if f:
            m("CuSegNOneM", thin(f["segment"]))
            m("CuFrontMaxOneM", str(f["front_max"]))
            m("CuFrontMinOneM", str(f["front_min"]))
    p = RES / "curve_ranges.json"
    if p.exists():
        j = json.loads(p.read_text())
        m("CuRangeEst", f3(j["estimated"]))
        m("CuRangeExact", f3(j["exact_ranges"]))
    p = RES / "static_curves.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["param"] = d["param"].fillna("")

    def get(n, method, param="", col="recall"):
        r = d[(d.n == n) & (d.method == method) & (d.param == param)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    sizes = sorted(d.n.unique())
    big, small = sizes[-1], sizes[0]
    ours = lambda ef, W=24: f"R=8,ef={ef},W={W}h"   # noqa: E731
    rows = [("Exact scan, fitted curve ($R{=}8$)", "exact scan, fitted curve", "R=8"),
            ("Folded HNSW, $ef{=}64$ (ours)", "folded (ours)", ours(64)),
            ("Folded HNSW, $ef{=}256$ (ours)", "folded (ours)", ours(256)),
            ("Folded HNSW, $ef{=}64$, $W{=}6$\\,h (ours)", "folded (ours)", ours(64, 6)),
            ("Overfetch + re-rank, $K{=}100$", "overfetch + re-rank", "K=100"),
            ("Overfetch + re-rank, $K{=}1000$", "overfetch + re-rank", "K=1000"),
            ("Relevance only", "relevance only", "")]
    lines = [r"\begin{tabular}{l rr rr}", r"\toprule",
             rf" & \multicolumn{{2}}{{c}}{{{thin(small)} memories}} & \multicolumn{{2}}{{c}}{{{thin(big)} memories}}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & recall@10 & ms & recall@10 & ms\\", r"\midrule"]
    for lab, meth, par in rows:
        cells = []
        for n in (small, big):
            rc, ms = get(n, meth, par), get(n, meth, par, "ms")
            cells += ["--" if math.isnan(rc) else f3(rc),
                      "--" if math.isnan(ms) else (f"{ms:.0f}" if ms >= 10 else f2(ms))]
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_curves_static.tex").write_text("\n".join(lines) + "\n")
    for n, tag in ((small, "HundredK"), (big, "OneM")):
        m(f"CuFitRecall{tag}", f3(get(n, "exact scan, fitted curve", "R=8")))
        m(f"CuFitMs{tag}", f"{get(n, 'exact scan, fitted curve', 'R=8', 'ms'):.0f}")
        m(f"CuOursRecall{tag}", f3(get(n, "folded (ours)", ours(64))))
        m(f"CuOursMs{tag}", f2(get(n, "folded (ours)", ours(64), "ms")))
        m(f"CuOursDeep{tag}", f3(get(n, "folded (ours)", ours(256))))
        m(f"CuOursDeepMs{tag}", f2(get(n, "folded (ours)", ours(256), "ms")))
        m(f"CuRerank{tag}", f3(get(n, "overfetch + re-rank", "K=100")))
        m(f"CuRerankK{tag}", f3(get(n, "overfetch + re-rank", "K=1000")))
        m(f"CuRerankKMs{tag}", f2(get(n, "overfetch + re-rank", "K=1000", "ms")))
        m(f"CuRel{tag}", f3(get(n, "relevance only")))
        r = d[(d.n == n) & (d.method == "paper score vs ACT-R score")]
        if len(r):
            m(f"CuOverlap{tag}", pct(float(r.recall.iloc[0])))
            nums = [float(x) for x in __import__("re").findall(r"([0-9.]+) h", r.extra.iloc[0])]
            if len(nums) == 2:
                m(f"CuAgeActr{tag}", f"{nums[0]:.1f}")
                m(f"CuAgePaper{tag}", f"{nums[1]:.1f}")
        r = d[(d.n == n) & (d.method == "folded (ours)") & (d.param == ours(64))]
        if len(r):
            m(f"CuRecent{tag}", thin(int(r.extra.iloc[0].split()[2].replace(",", ""))))
    m("CuOursSixRecallOneM", f3(get(big, "folded (ours)", ours(64, 6))))
    m("CuOursSixMsOneM", f2(get(big, "folded (ours)", ours(64, 6), "ms")))
    r = d[(d.n == big) & (d.method == "folded (ours)") & (d.param == ours(64, 6))]
    if len(r):
        m("CuRecentSixOneM", thin(int(r.extra.iloc[0].split()[2].replace(",", ""))))
    for R, tag in ((4, "Four"), (6, "Six"), (8, "Eight"), (10, "Ten")):
        m(f"CuR{tag}", f3(get(small, "folded (ours)", f"R={R},ef=128,W=24h")))
    m("CuSpeedupOneM", f"{get(big, 'exact scan, fitted curve', 'R=8', 'ms') / get(big, 'folded (ours)', ours(64), 'ms'):.0f}")


def curves_town() -> None:
    p = RES / "town_curve.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    outcomes = ("aware_at_party", "recalled_at_plan", "attended")
    # paired Wilcoxon against the exact scan of the same score, Holm-corrected over this table
    tests = []
    for (b, sc), db in d.groupby(["burn", "score"]):
        for key in ("folded", "rerank100"):
            for col in outcomes:
                piv = db.pivot_table(index="seed", columns="backend_key", values=col)
                if key not in piv or "exact" not in piv:
                    continue
                pv = piv[[key, "exact"]].dropna()
                diff = pv[key] - pv["exact"]
                pval = stats.wilcoxon(pv[key], pv["exact"]).pvalue if (diff != 0).any() else 1.0
                tests.append(((b, sc, key, col), pval))
    holm, running = {}, 0.0
    for rank, (key, pval) in enumerate(sorted(tests, key=lambda x: x[1])):
        running = max(running, min(1.0, (len(tests) - rank) * pval))
        holm[key] = running
    order = [("exact", "Exact scan"), ("folded", "Folded HNSW (ours)"), ("rerank100", "Overfetch, $K{=}100$")]
    scores = [("last", "last use"), ("uses", "every use"), ("uses_log", "every use, log")]
    burns = sorted(d.burn.unique())
    lines = [r"\begin{tabular}{l l l r r r r r}", r"\toprule",
             r"Run & Curve & Retrieval & memories & recall & knew & recalled & attended\\", r"\midrule"]
    for b in burns:
        for si, (sc, slab) in enumerate(scores):
            for i, (key, lab) in enumerate(order):
                dk = d[(d.burn == b) & (d.score == sc) & (d.backend_key == key)]
                if dk.empty:
                    continue
                cells = []
                for col in outcomes:
                    n = len(dk)
                    mean = dk[col].mean()
                    ci = stats.t.ppf(0.975, n - 1) * dk[col].std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                    pv = holm.get((b, sc, key, col), 1.0)
                    mark = "$^{**}$" if pv < 0.01 else ("$^{*}$" if pv < 0.05 else "")
                    cells.append(f"{mean:.1f} $\\pm$ {ci:.1f}{mark}")
                run = f"{b + 2} days" if (si == 0 and i == 0) else ""
                curve = slab if i == 0 else ""
                lines.append(f"{run} & {curve} & {lab} & {dk.mean_memories.mean():.0f} & {dk.recall.mean():.3f} & "
                             + " & ".join(cells) + r"\\")
            if si < len(scores) - 1 and not d[(d.burn == b) & (d.score == scores[si + 1][0])].empty:
                lines.append(r"\addlinespace[2pt]")
        if b != burns[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_curves_town.tex").write_text("\n".join(lines) + "\n")
    # numbers for the text
    for b in burns:
        btag = {30: "Thirty", 90: "Ninety"}.get(int(b))
        if btag is None:
            continue
        for sc, stag in (("last", "Last"), ("uses", "Uses"), ("uses_log", "Log")):
            db = d[(d.burn == b) & (d.score == sc)]
            if db.empty:
                continue
            piv = {c: db.pivot_table(index="seed", columns="backend_key", values=c) for c in outcomes + ("recall",)}
            for key, ktag in (("exact", "Ex"), ("folded", "Fo"), ("rerank100", "Re")):
                if key not in piv["attended"]:
                    continue
                for col, ctag in zip(outcomes, ("Aware", "Recalled", "Att")):
                    m(f"Tc{btag}{stag}{ktag}{ctag}", f"{piv[col][key].mean():.1f}")
                m(f"Tc{btag}{stag}{ktag}Recall", f3(piv["recall"][key].mean()))
                m(f"Tc{btag}{stag}{ktag}RecallFour", f"{piv['recall'][key].mean():.4f}")
            if "folded" in piv["attended"] and "exact" in piv["attended"]:
                both = pd.concat([piv[c][["folded", "exact"]] for c in outcomes], axis=1).dropna()
                same = np.logical_and.reduce([(piv[c]["folded"] == piv[c]["exact"]).reindex(both.index) for c in outcomes])
                maxdiff = max(float((piv[c]["folded"] - piv[c]["exact"]).abs().max()) for c in outcomes)
                m(f"Tc{btag}{stag}FoSame", str(int(np.sum(same))))
                m(f"Tc{btag}{stag}Seeds", str(len(both)))
                m(f"Tc{btag}{stag}FoMaxDiff", f"{maxdiff:.0f}")
            for col, ctag in zip(outcomes, ("Aware", "Recalled", "Att")):
                m(f"Tc{btag}{stag}Re{ctag}Holm", f"{holm.get((b, sc, 'rerank100', col), 1.0):.2g}")
                m(f"Tc{btag}{stag}Fo{ctag}Holm", f"{holm.get((b, sc, 'folded', col), 1.0):.2g}")
        db = d[(d.burn == b) & (d.backend_key == "folded")]
        if len(db) and btag:
            m(f"Tc{btag}FoMs", f2(db.search_ms.mean()))
        de = d[(d.burn == b) & (d.backend_key == "exact")]
        if len(de) and btag:
            m(f"Tc{btag}ExMs", f2(de.search_ms.mean()))
    m("TcRuns", thin(len(d)))
    # shape of the two curves after min-max scaling over 92 days of memories (for the text)
    from agentmem import ACT_R, exponential
    span = 92 * 24.0
    for name, curve in (("Exp", exponential(0.995)), ("Actr", ACT_R)):
        lo, hi = float(curve(span)), float(curve(0.0))
        for h, htag in ((5.0, "Five"), (35.0, "ThirtyFive")):
            m(f"TcShape{name}{htag}", f"{(float(curve(h)) - lo) / (hi - lo):.2f}")
    # the curve itself: exact retrieval with ACT-R's curve against the paper's exponential on the same
    # seeds (same world), paired Wilcoxon, Holm-corrected over these comparisons
    pt = RES / "town.csv"
    if not pt.exists():
        return
    base = pd.read_csv(pt)
    base = base[(base.profile == "paper") & (base.backend_key == "exact")]
    vs, cells = [], {}
    for b in burns:
        btag = {30: "Thirty", 90: "Ninety"}.get(int(b))
        pb = base[base.burn == b].set_index("seed")
        if btag is None or pb.empty:
            continue
        for sc, stag in (("last", "Last"), ("uses", "Uses"), ("uses_log", "Log")):
            dc = d[(d.burn == b) & (d.score == sc) & (d.backend_key == "exact")].set_index("seed")
            common = dc.index.intersection(pb.index)
            if len(common) < 2:
                continue
            for col, ctag in zip(outcomes, ("Aware", "Recalled", "Att")):
                x, y = dc.loc[common, col], pb.loc[common, col]
                pval = stats.wilcoxon(x, y).pvalue if (x != y).any() else 1.0
                vs.append(((btag, stag, ctag), pval))
                cells[(btag, ctag)] = f"{y.mean():.1f}"
                m(f"Tc{btag}{stag}VsExpSeeds", str(len(common)))
    running = 0.0
    for rank, (key, pval) in enumerate(sorted(vs, key=lambda x: x[1])):
        running = max(running, min(1.0, (len(vs) - rank) * pval))
        m(f"Tc{key[0]}{key[1]}VsExp{key[2]}Holm", f"{running:.2g}")
    for (btag, ctag), v in cells.items():
        m(f"Tc{btag}Exp{ctag}", v)
    # why counting every use locks the memory in (experiments/town_curve_diag.py)
    pdg = RES / "town_curve_diag.json"
    if pdg.exists():
        sm = json.loads(pdg.read_text())["summary"]
        for v, tag in (("last", "Last"), ("uses", "Uses"), ("uses_log", "Log")):
            if v not in sm:
                continue
            m(f"TcDiag{tag}MaxRet", thin(round(sm[v]["max_retrievals"])))
            m(f"TcDiag{tag}TopShare", pct(sm[v]["top1_share"]))
            m(f"TcDiag{tag}NewHour", f"{sm[v]['new_hour']:.2f}" if sm[v]["new_hour"] >= 0.01 else f"{sm[v]['new_hour']:.3f}")
            m(f"TcDiag{tag}PerMem", f"{sm[v]['retrievals_per_memory']:.1f}")
        m("TcDiagSeeds", str(sm["uses"]["seeds"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static()
    relsweep()
    normef_table()
    hnsw()
    langchain_and_reference()
    town()
    curves()
    curves_town()
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (OUT / "numbers.tex").write_text("% generated by experiments/tables.py\n" + "\n".join(lines) + "\n")
    print(f"{len(macros)} macros -> {OUT / 'numbers.tex'}")


if __name__ == "__main__":
    main()
