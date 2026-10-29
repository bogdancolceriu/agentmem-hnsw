"""Tur ghidat al proiectului: fiecare idee, rulată pe date mici, cu explicații.

    python demo/tour.py           # pașii 1-5, cam două minute, fără nicio descărcare
    python demo/tour.py --oras    # plus pasul 6, două simulări scurte ale orașului
                                  # (prima dată descarcă vectorii GloVe, cam 130 MB)

Fiecare pas spune ce arată, ce a ieșit și în ce fișier e codul. Experimentele complete
din raport sunt în experiments/ și se rulează cu comenzile din README.md. Datele de aici
sunt sintetice (vectori aleatori grupați pe teme), deci cifrele seamănă cu cele din
raport, dar nu sunt aceleași.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentmem import (ACT_R, PROFILES, CurveExact, CurveFolded, CurveOverfetch, ExactStore,  # noqa: E402
                      FoldedHNSW, OverfetchRerank, RelevanceOnlyHNSW, exact_scores, exponential,
                      fit_expsum, fold_memories, fold_query, minmax, ranges, tie_aware_recall, topk)
from agentmem.curve_stores import pareto_max  # noqa: E402
from agentmem.paper_hnsw import IP, PaperHNSW  # noqa: E402

DAYS = 90
NOW = DAYS * 24 + 0.25          # ora la care agentul își caută amintirile
W = PROFILES["paper"]          # scorul din articol: min-max, ponderi 1, 1, 1


def title(n: int, text: str) -> None:
    print(f"\n{'=' * 78}\nPasul {n}. {text}\n{'=' * 78}", flush=True)


def say(text: str) -> None:
    print(textwrap.fill(" ".join(text.split()), 78), flush=True)


def code(text: str) -> None:
    print(textwrap.fill("Codul: " + " ".join(text.split()), 78, subsequent_indent="       "), flush=True)


def unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def memories(n: int, d: int = 64, topics: int = 300, nq: int = 100, seed: int = 0):
    """n amintiri: embedding-uri grupate pe teme, momente de acces pe 90 de zile, importanță."""
    rng = np.random.default_rng(seed)
    C = unit(rng.normal(size=(topics, d)))
    E = unit(C[rng.integers(topics, size=n)] + rng.normal(size=(n, d)) / np.sqrt(d))
    Q = unit(C[rng.integers(topics, size=nq)] + rng.normal(size=(nq, d)) / np.sqrt(d))
    created = np.sort(rng.uniform(0.0, DAYS * 24.0, n))
    t_last = created.copy()
    again = rng.random(n) < 0.05                       # 5% au fost regăsite mai târziu
    t_last[again] = rng.uniform(created[again], DAYS * 24.0)
    vals = np.round(np.arange(1, 11) / 10, 1)          # importanța 1..10, scalată la 0,1..1
    p = np.array([30, 25, 15, 10, 6, 5, 4, 2, 2, 1], dtype=float)
    imp = rng.choice(vals, size=n, p=p / p.sum())
    return E, imp, created, t_last, Q


def num(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def age(h: float) -> str:
    if h < 1:
        return "sub o oră"
    return f"{h:.0f} ore" if h < 48 else f"{h / 24:.0f} zile"


# ---------------------------------------------------------------------------------------- 1
def step_hnsw() -> None:
    title(1, "HNSW din articol, scris după pseudocod, comparat cu biblioteca autorilor")
    say("""Căutăm cei 10 vecini cei mai apropiați ai unei interogări printre 20 000 de vectori
        cu 32 de dimensiuni. Scanarea exactă calculează 20 000 de distanțe. HNSW pornește din
        nivelul de sus al grafului, coboară lacom și pe nivelul 0 ține o listă de ef candidați.
        Prima rulare compilează codul cu numba, deci durează 10-30 de secunde.""")
    rng = np.random.default_rng(1)
    n, d, nq = 20000, 32, 200
    X, Q = unit(rng.normal(size=(n, d))), unit(rng.normal(size=(nq, d)))
    gt = np.argsort(-(Q @ X.T), axis=1)[:, :10]
    t0 = time.perf_counter()
    ours = PaperHNSW(M=16, ef_construction=100, metric=IP).build(X)
    print(f"  graful nostru: {time.perf_counter() - t0:.1f} s (cu compilarea), {ours.max_level + 1} niveluri")
    import hnswlib
    ref = hnswlib.Index(space="ip", dim=d)
    ref.init_index(max_elements=n, ef_construction=100, M=16, random_seed=0)
    ref.add_items(X, np.arange(n), num_threads=1)
    print("                ef   recall@10, al nostru   recall@10, hnswlib   distanțe calculate")
    for ef in (16, 64, 256):
        lab, dist, _ = ours.search(Q, 10, ef)
        ref.set_ef(ef)
        lab_ref, _ = ref.knn_query(Q, k=10)
        r1 = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(lab, gt)])
        r2 = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(lab_ref, gt)])
        print(f"  {ef:17d}   {r1:20.3f}   {r2:18.3f}   {num(round(dist)):>8} din {num(n)} ({100 * dist / n:.1f}%)")
    say("""Cu ef mai mare găsești mai mulți vecini adevărați, dar calculezi mai multe distanțe.
        Implementarea noastră și hnswlib ajung la aproape același recall, pentru că urmează
        același algoritm. Datele aleatoare uniforme sunt cazul greu, pe date reale recall-ul
        e mai mare la același ef.""")
    code("agentmem/paper_hnsw.py (Algoritmii 1-5). Studiul complet: experiments/exp_hnsw_ablation.py")


# ---------------------------------------------------------------------------------------- 2
def step_score(E, imp, t_last, q) -> np.ndarray:
    title(2, "Scorul cu care își aleg amintirile agenții generativi")
    say(f"""{num(len(E))} de amintiri pe {DAYS} de zile. Scorul din articol adună trei termeni,
        fiecare scalat la [0, 1]: recența (0,995 la puterea orelor de la ultimul acces),
        importanța și relevanța (similaritatea cu întrebarea).""")
    s = exact_scores(E, imp, t_last, q, NOW, W)
    sim = E @ q
    sim_rank = np.empty(len(E), dtype=np.int64)
    sim_rank[np.argsort(-sim)] = np.arange(1, len(E) + 1)
    rec, im, rel = minmax(0.995 ** (NOW - t_last)), minmax(imp), minmax(sim)
    print("  loc     vechime  import.  similar.   rec. + imp. + rel. = scor   locul după similaritate")
    for i, m in enumerate(topk(s, 10), 1):
        print(f"  {i:3d}  {age(NOW - t_last[m]):>10}  {imp[m] * 10:7.0f}  {sim[m]:8.2f}   "
              f"{rec[m]:.2f} + {im[m]:.2f} + {rel[m]:.2f} = {s[m]:.2f}   {num(int(sim_rank[m])):>12}")
    outside = int(np.sum(sim_rank[topk(s, 10)] > 100))
    say(f"""Ultima coloană arată de ce nu ajunge căutarea vectorială obișnuită: {outside} din cele 10
        amintiri alese de scor nu sunt nici măcar printre primele 100 după similaritate. Un index
        vectorial știe doar similaritatea, deci LangChain ia primele 100 după similaritate și le
        reordonează. Pe acestea {outside} nu le-ar evalua niciodată.""")
    code("agentmem/score.py, funcția exact_scores")
    return s


# ---------------------------------------------------------------------------------------- 3
def step_fold(E, imp, t_last, q, s) -> None:
    title(3, "Ideea extensiei: tot scorul e un singur produs scalar")
    say("""Recența se desparte în jurul unei ancore τ: 0,995^(T-t) = 0,995^(T-τ) · 0,995^(τ-t).
        Al doilea factor ține doar de amintire, primul doar de momentul întrebării. Amintirea
        primește vectorul [e, 0,995^(τ-t), importanța], iar întrebarea [c·q, a·0,995^(T-τ), b].
        Normalizarea min-max doar rescalează ponderile a, b, c.""")
    wr = W.raw(*ranges(E, imp, t_last, q, NOW))
    anchor = float(t_last.max())
    X = fold_memories(E, imp, t_last, anchor)
    Qv = fold_query(q, NOW, anchor, wr)
    ip = X.astype(np.float64) @ Qv.astype(np.float64)
    shift = ip - s
    err = float(np.abs(shift - np.median(shift)).max())
    same = list(topk(ip, 10)) == list(topk(s, 10))
    print(f"  vectorul unei amintiri: {E.shape[1]} + 2 = {X.shape[1]} numere")
    print(f"  produs scalar - scor = aceeași constantă pentru toate amintirile, cu abatere de cel mult {err:.0e}")
    print(f"  top-10 după produsul scalar = top-10 după scor: {'da' if same else 'nu'}")
    say("""Constanta vine din min-max, care scade minimul fiecărui termen. E aceeași pentru toate
        amintirile, deci nu schimbă clasamentul. Așa că top-10 după scor e o căutare după cel
        mai mare produs scalar (MIPS), pe care HNSW o face direct.""")
    code("agentmem/score.py, funcțiile fold_memories și fold_query")


# ---------------------------------------------------------------------------------------- 4
def step_stores(E, imp, t_last, Q) -> None:
    title(4, "Patru moduri de regăsire, pe aceleași amintiri")
    say(f"""Aceleași {num(len(E))} de amintiri, {len(Q)} de întrebări. Pentru fiecare metodă: ce
        fracțiune din adevăratul top-10 găsește (recall) și cât durează o întrebare. Construirea
        celor trei indexuri HNSW durează câteva zeci de secunde.""")
    n, d = E.shape
    stores = [("scanare exactă (referința)", ExactStore(d, capacity=n)),
              ("indexul pliat (al nostru)", FoldedHNSW(d, buffer_size=n + 1, capacity=n)),
              ("overfetch, K = 100 (LangChain)", OverfetchRerank(d, fetch=100, capacity=n)),
              ("doar relevanța (RAG simplu)", RelevanceOnlyHNSW(d, capacity=n))]
    for _, st in stores:
        st.add(E, imp, t_last)
        if isinstance(st, FoldedHNSW):
            st.flush(float(t_last.max()))
    print("  metodă                            recall@10   ms pe întrebare")
    for name, st in stores:
        rec, ms = [], []
        for q in Q:
            truth = exact_scores(E, imp, t_last, q, NOW, W)
            t0 = time.perf_counter()
            ids = st.retrieve(q, NOW, 10, W, refresh=False)
            ms.append(1000 * (time.perf_counter() - t0))
            rec.append(tie_aware_recall(ids, truth, 10))
        print(f"  {name:32s}  {np.mean(rec):9.3f}   {np.median(ms):9.2f}")
    say("""Indexul pliat găsește aproape tot ce găsește scanarea exactă. Ce lipsește vine din
        aproximarea HNSW și din estimarea amplitudinii relevanței. Overfetch-ul vede doar cele
        100 de amintiri cele mai asemănătoare (plus cele mai noi), deci ratează aproape tot.
        La memorii mici scanarea exactă e la fel de rapidă. Diferența de timp apare de la zeci
        de mii de amintiri în sus (în raport, la un milion: 0,53 ms față de 46-65 ms).""")
    code("""agentmem/stores.py (ExactStore, FoldedHNSW, OverfetchRerank, RelevanceOnlyHNSW).
         Experimentul complet: experiments/exp_static.py""")


# ---------------------------------------------------------------------------------------- 5
def step_curves() -> None:
    title(5, "A doua extensie: curba de uitare din ACT-R, tot ca produs scalar")
    f = fit_expsum(ACT_R, 8)
    rel, of_range = f.errors()
    paper = exponential(0.995)
    say("""În ACT-R, fiecare folosire a unei amintiri contează, cu o pondere care scade ca o
        putere a timpului: (1 + vechime/1 h)^(-0,5). Puterea nu se desparte în jurul unei ancore,
        dar e aproape exact o sumă de 8 exponențiale, iar fiecare exponențială se desparte.""")
    print("  vechimea unei folosiri    ACT-R    suma de 8 exponențiale    exponențiala din articol")
    for h, lab in ((1, "1 oră"), (24, "1 zi"), (168, "1 săptămână"), (720, "1 lună"), (8760, "1 an")):
        print(f"  {lab:22s}  {ACT_R(h):7.4f}   {f(h):22.4f}   {paper(h):23.4f}")
    print(f"  eroarea cea mai mare pe un an: {100 * of_range:.2f}% din amplitudinea curbei")

    rng = np.random.default_rng(2)
    n, d = 20000, 64
    E, imp, created, _, Q = memories(n, d, nq=50, seed=3)
    uses = rng.geometric(0.45, n) - 1                     # 1,2 folosiri în plus, în medie
    hist = [np.concatenate([[c], np.sort(rng.uniform(c, DAYS * 24.0, u))]) for c, u in zip(created, uses)]
    say(f"""Acum {num(n)} de amintiri cu istorie de folosire: crearea și, în medie, încă 1,2
        folosiri. Scorul: suma ACT-R peste toate folosirile, plus importanța și relevanța,
        scalate min-max.""")
    exact, folded = CurveExact(d), CurveFolded(d, exact_ranges_below=0)
    over = CurveOverfetch(d, fetch=100)
    for st in (exact, folded, over):
        st.load(E, imp, hist, NOW)
    seg = max(folded.segments, key=lambda sg: len(sg["h"].live))
    hi, lo, _ = folded._fronts(seg)
    print(f"  partea recentă (folosite în ultimele 24 de ore, scanate exact): {len(folded.recent)} amintiri")
    print(f"  în segment: {len(seg['h'].live)} amintiri, dintre care pe fronturile Pareto: {len(hi)} (maxim), {len(lo)} (minim)")
    print("  metodă                                recall@10   ms pe întrebare")
    for name, st in (("indexul pliat, 8 exponențiale", folded), ("overfetch, K = 100", over)):
        rec, ms = [], []
        for q in Q:
            truth = exact.exact(q, NOW, W)
            t0 = time.perf_counter()
            ids = st.retrieve(q, NOW, 10, W, refresh=False)
            ms.append(1000 * (time.perf_counter() - t0))
            rec.append(tie_aware_recall(ids, truth, 10))
        print(f"  {name:36s}  {np.mean(rec):9.3f}   {np.median(ms):9.2f}")
    same = np.mean([len(set(topk(exact.exact(q, NOW, W), 10)) & set(topk(exact_scores(
        E, imp, np.array([h[-1] for h in hist]), q, NOW, W), 10))) / 10 for q in Q])
    say(f"""Amplitudinea termenului de recență vine exact din fronturile Pareto, care sunt mici,
        deci nu costă aproape nimic. Curba chiar schimbă ce își amintește agentul: doar
        {100 * same:.0f}% din top-10 cu ACT-R e și în top-10 cu exponențiala din articol.""")
    code("agentmem/curves.py (aproximarea), agentmem/curve_stores.py (CurveFolded). "
         "Experimentele complete: experiments/exp_curves.py și experiments/exp_town_curve.py")


# ---------------------------------------------------------------------------------------- 6
def step_town() -> None:
    title(6, "Orașul Willowbrook: aceeași petrecere, cu două moduri de regăsire")
    say("""25 de agenți pe reguli trăiesc 9 zile. În ultimele două, Maria află că dă o petrecere
        și îi invită pe cei cu care vorbește. Aceeași lume (același seed), o dată cu scorul
        exact din articol și o dată doar cu relevanța. Fiecare simulare durează câteva secunde.""")
    import pandas as pd
    from town.sim import SimConfig, Town, load_encoder
    d = pd.read_csv(ROOT / "results" / "town.csv")
    d = d[(d.burn == 7) & (d.profile == "paper")]
    p = d.pivot_table(index="seed", columns="backend_key", values="aware_at_party").dropna()
    diff = p["exact"] - p["relonly"]
    seed = int((diff - diff.mean()).abs().idxmin())      # un seed tipic, nu cel mai spectaculos
    enc = load_encoder()
    print(f"  seed {seed}           au aflat de petrecere   și-au amintit-o la planificare   au venit")
    for kind, lab in (("exact", "scor exact"), ("relonly", "doar relevanța")):
        cfg = SimConfig(days=9, party_day=8, seed=seed, backend=kind, weights=W)
        row = Town(cfg, encoder=enc).run()
        print(f"  {lab:15s}  {int(row['aware_at_party']):15d} din 25   {int(row['recalled_at_plan']):23d}   "
              f"{int(row['attended']):9d}")
    say("""Invitația circulă prin conversații, iar într-o conversație agentul își caută amintirile
        după numele celuilalt. Fără recență și importanță petrecerea rămâne îngropată, deci mai
        puțini află de ea. În raport sunt câte 30 de seed-uri pentru fiecare variantă. Pentru o
        reluare pas cu pas a orașului, deschide demo/willowbrook.html în browser.""")
    code("town/world.py (lumea), town/sim.py (bucla agenților). Experimentul complet: experiments/exp_town.py")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oras", action="store_true", help="rulează și simulările orașului (pasul 6)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    step_hnsw()
    E, imp, _, t_last, Q = memories(50000)
    s = step_score(E, imp, t_last, Q[0])
    step_fold(E, imp, t_last, Q[0], s)
    step_stores(E, imp, t_last, Q)
    step_curves()
    if args.oras:
        step_town()
    else:
        print("\n(Pasul 6, orașul, rulează cu: python demo/tour.py --oras)")
    print(f"\nGata în {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    main()
