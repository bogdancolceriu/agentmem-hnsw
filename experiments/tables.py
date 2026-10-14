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
        m("LcOursRecall", f3(j["recall_ours"]))
        m("LcMinN", thin(min(a["memories"] for a in j["agents"])))
        m("LcMaxN", thin(max(a["memories"] for a in j["agents"])))
    p = RES / "ga_reference_check.json"
    if p.exists():
        j = json.loads(p.read_text())["town_summary"]
        m("GaOverlap", pct(j["overlap_reference_paper"]))
        m("GaAgeRef", f"{j['age_reference']:.1f}")
        m("GaAgePaper", f"{j['age_paper']:.1f}")


# ------------------------------------------------------------------ town

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
             r"Age & Retrieval & memories & recall & knew & recalled & attended\\", r"\midrule"]
    for b in burns:
        db = d[d.burn == b]
        ex = db[db.backend_key == "exact"].set_index("seed")
        for i, (key, lab) in enumerate(order):
            dk = db[db.backend_key == key].set_index("seed")
            if dk.empty:
                continue
            cells = []
            for col in ("aware_at_party", "recalled_at_plan", "attended"):
                mean, ci = dk[col].mean(), 1.96 * dk[col].std(ddof=1) / np.sqrt(len(dk))
                mark = ""
                if key != "exact":
                    common = ex.index.intersection(dk.index)
                    diff = (dk.loc[common, col] - ex.loc[common, col]).values
                    if len(common) > 5 and np.any(diff != 0):
                        pv = stats.wilcoxon(dk.loc[common, col], ex.loc[common, col]).pvalue
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static()
    relsweep()
    normef_table()
    hnsw()
    langchain_and_reference()
    town()
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (OUT / "numbers.tex").write_text("% generated by experiments/tables.py\n" + "\n".join(lines) + "\n")
    print(f"{len(macros)} macros -> {OUT / 'numbers.tex'}")


if __name__ == "__main__":
    main()
