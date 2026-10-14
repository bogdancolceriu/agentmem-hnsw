"""LLM-in-the-loop validation (runs on a laptop GPU through Ollama).

The town is simulated with rules during the burn-in. From the moment Maria
Lopez learns about her party (35 hours before it starts) two decisions are made
by a local language model instead of rules, from the memories that the
agent's retrieval policy returns:

  conversations  the speaker sees the pieces of news among its retrieved
                 memories and the model chooses which one, if any, to mention
  planning       the agent sees all memories retrieved for its plan (the focal
                 points of plan.py) and the model writes the events it will
                 attend today; an event counts only if it matches news the
                 agent actually retrieved

The retrieval policies are run without approximate search here (exact scans),
so that only the policy differs: the paper's full score (what our folded HNSW
index returns), LangChain-style overfetch+rerank, and relevance only.

Requirements: Python 3.10+, numpy, Ollama (https://ollama.com) with a model, e.g.
    ollama pull qwen2.5:7b
    python -m experiments.llm_validation --model qwen2.5:7b
Results: results/llm_validation.csv, results/llm_validation_log.jsonl
Use --mock to test the pipeline without a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from agentmem import PROFILES
from town.sim import KIND_NEWS, SimConfig, Town, load_encoder
from town.world import date_str

ROOT = Path(__file__).resolve().parents[1]
STOP = set("a an the and or of on at in to for is are was be will with by from about that this it as "
           "pm am me my i told everyone invited".split())


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+|\d+", s.lower()) if t not in STOP}


class LLM:
    def __init__(self, model: str, host: str, mock: bool, log_path: Path):
        self.model, self.host, self.mock = model, host.rstrip("/"), mock
        self.log = log_path.open("a", encoding="utf-8")
        self.calls, self.seconds = 0, 0.0

    def ask(self, system: str, user: str, tag: dict) -> dict:
        t0 = time.perf_counter()
        if self.mock:
            out = self._mock(user)
        else:
            body = json.dumps(dict(model=self.model, stream=False, format="json",
                                   messages=[dict(role="system", content=system), dict(role="user", content=user)],
                                   options=dict(temperature=0, seed=0, num_ctx=4096))).encode()
            req = urllib.request.Request(self.host + "/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                content = json.loads(r.read())["message"]["content"]
            try:
                out = json.loads(content)
            except json.JSONDecodeError:
                out = {}
        dt = time.perf_counter() - t0
        self.calls += 1
        self.seconds += dt
        self.log.write(json.dumps(dict(tag, system=system, user=user, answer=out, seconds=round(dt, 3))) + "\n")
        self.log.flush()
        return out if isinstance(out, dict) else {}

    @staticmethod
    def _mock(user: str) -> dict:
        if "Which one" in user:
            nums = re.findall(r"^(\d+)\. ", user, flags=re.M)
            h = zlib.crc32(user.encode()) % 3
            return {"choice": int(nums[0]) if nums and h else 0}
        today = re.search(r"Today is (\w+, \w+ \d+)", user).group(1)
        ev = [dict(what=line, place="", start="") for line in user.splitlines()
              if today in line and ("party" in line or "picnic" in line)]
        return {"events": ev[:2]}


class LLMTown(Town):
    def __init__(self, cfg: SimConfig, llm: LLM, encoder=None):
        super().__init__(cfg, encoder=encoder)
        self.llm = llm
        self.llm_from = self.news[self.party].announce

    def _persona(self, a) -> str:
        r = a.res
        job = f"works as a {r.role.replace('_', ' ')} at {r.work}" if r.role else "is retired"
        return (f"You are role-playing {a.name}, a resident of the small town of Willowbrook who {job}. "
                f"{a.name.split()[0]} is interested in {', '.join(r.interests)}. Answer only with JSON.")

    # ------------------------------------------------------------ planning
    def _plan(self, a, day, T, focal_points):
        if T < self.llm_from:
            return super()._plan(a, day, T, focal_points)
        ids = np.unique(np.concatenate([self._retrieve(a, f, T, self.cfg.k_plan, 1) for f in focal_points]))
        cands = {}
        for i in ids.tolist():
            if a.kind[i] == KIND_NEWS:
                nid = a.nid[i]
                if nid == self.party:
                    self.party_in_context.append((a.idx, T))
                n = self.news[nid]
                if n.day == day and nid not in a.planned and (T % 24.0) < n.end:
                    cands[nid] = n
        if self.party in cands or (self.party in a.hosting and self.news[self.party].day == day):
            self.plan_hits.setdefault(day, set()).add(a.idx)
        for nid in a.hosting:                         # hosts always go to their own event
            n = self.news[nid]
            if n.day == day and nid not in a.planned and (T % 24.0) < n.end:
                self._schedule(a, nid, T)
        order = sorted(ids.tolist(), key=lambda i: a.store.t_created[i])
        lines = "\n".join(f"- {a.text[i]}" for i in order)
        clock = f"{int(T % 24):02d}:{int(round((T % 1) * 60)):02d}"
        user = (f"Today is {date_str(day)} and it is {clock}. These are the statements {a.name} recalls:\n"
                f"{lines}\n\nGiven the statements above, is there anything {a.name} should remember as they "
                f"plan for today? List the events that {a.name} will attend later today, with place and start "
                f'time, as JSON: {{"events": [{{"what": "...", "place": "...", "start": "HH:MM"}}]}}. '
                f'If there are none, answer {{"events": []}}.')
        out = self.llm.ask(self._persona(a), user, dict(kind="plan", agent=a.name, T=T,
                                                         backend=self.cfg.backend, seed=self.cfg.seed,
                                                         party_in_memory=self.party in cands))
        for ev in (out.get("events") or [])[:5]:
            if not isinstance(ev, dict):
                continue
            words = tokens(" ".join(str(ev.get(x, "")) for x in ("what", "place", "start")))
            best, best_overlap = None, 1
            for nid, n in cands.items():
                ov = len(words & tokens(n.content))
                if ov > best_overlap:
                    best, best_overlap = nid, ov
            if best is not None and best not in a.planned:
                self._schedule(a, best, T)

    def _schedule(self, a, nid, T):
        n = self.news[nid]
        cur = int((T % 24.0) * 4)
        for s in range(max(int(n.start * 4), cur + 1), int(n.end * 4)):
            a.today[s] = (n.place, f"attending {n.title}")
        a.planned[nid] = int(T // 24)

    # ------------------------------------------------------------ conversations
    def _news_to_tell(self, sp, li, T):
        if T < self.llm_from:
            return super()._news_to_tell(sp, li, T)
        day = int(T // 24)
        for nid in sp.hosting:
            n = self.news[nid]
            if (li.idx, nid) not in sp.told and (n.day is None or n.day >= day):
                return nid
        act = li.today[int((T % 24.0) * 4)][1]
        ids = np.union1d(self._retrieve(sp, li.name, T, self.cfg.k_conv1, 0),
                         self._retrieve(sp, f"{li.name} is {act}", T, self.cfg.k_conv2, 0))
        cands = {}
        for i in ids.tolist():
            if sp.kind[i] != KIND_NEWS:
                continue
            nid = sp.nid[i]
            n = self.news[nid]
            if (li.idx, nid) in sp.told or nid in cands:
                continue
            if n.day is not None and (n.day < day or (n.day == day and (T % 24.0) >= n.end)):
                continue
            cands[nid] = n
        if not cands:
            return None
        items = sorted(cands.items(), key=lambda kv: -kv[1].importance)[:8]
        listing = "\n".join(f"{j}. {n.content}" for j, (_, n) in enumerate(items, 1))
        place = sp.today[int((T % 24.0) * 4)][0]
        user = (f"{sp.name} runs into {li.name} at {place}; {li.name} is {act}. These are pieces of news "
                f"{sp.name} remembers:\n{listing}\n\nWhich one of them, if any, would {sp.name} naturally "
                f'mention to {li.name} in a short chat? Answer as JSON: {{"choice": <number>}}, or '
                f'{{"choice": 0}} to just make small talk.')
        out = self.llm.ask(self._persona(sp), user, dict(kind="talk", agent=sp.name, listener=li.name, T=T,
                                                          backend=self.cfg.backend, seed=self.cfg.seed,
                                                          party_offered=self.party in cands))
        try:
            c = int(out.get("choice", 0))
        except (TypeError, ValueError):
            c = 0
        return items[c - 1][0] if 1 <= c <= len(items) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--burn", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policies", nargs="+", default=["exact", "rerank_exact", "relonly_exact"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_validation.csv"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.model, args.host, args.mock, out.with_name(out.stem + "_log.jsonl"))
    enc = load_encoder()
    fields = ["model", "policy", "seed", "burn", "aware_at_party", "recalled_at_plan", "planned", "attended",
              "llm_calls", "llm_seconds", "mean_memories"]
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for seed in range(args.seeds):
            for policy in args.policies:
                kw = {"fetch": 100} if policy.startswith("rerank") else {}
                cfg = SimConfig(days=args.burn + 2, party_day=args.burn + 1, seed=seed, backend=policy,
                                backend_kw=kw, weights=PROFILES["paper"], measure_recall=False)
                c0, s0 = llm.calls, llm.seconds
                t0 = time.time()
                town = LLMTown(cfg, llm, encoder=enc)
                res = town.run()
                row = dict(model="mock" if args.mock else args.model, policy=policy, seed=seed, burn=args.burn,
                           aware_at_party=res["aware_at_party"], recalled_at_plan=res["recalled_at_plan"],
                           planned=res["planned"], attended=res["attended"], llm_calls=llm.calls - c0,
                           llm_seconds=round(llm.seconds - s0, 1), mean_memories=round(res["mean_memories"]))
                w.writerow(row)
                f.flush()
                print(f"seed {seed} {policy:14s} aware={row['aware_at_party']:2d} recalled={row['recalled_at_plan']:2d} "
                      f"attended={row['attended']:2d}  ({row['llm_calls']} LLM calls, {time.time() - t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
