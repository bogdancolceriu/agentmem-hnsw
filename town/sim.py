"""Closed-loop simulation of rule-based generative agents in Willowbrook.

The cognitive loop mirrors Generative Agents (Park et al., 2023) with the LLM
calls replaced by deterministic rules, so that the only thing that differs
between two runs with the same seed is the memory store:

  perceive  every 15 min, each resident observes up to `att_bandwidth` people
            at the same place; new observations enter the memory stream with a
            rule-based importance (instead of an LLM poignancy rating)
  converse  co-located residents talk; before talking, the speaker retrieves
            memories with the same focal points as converse.py in the reference
            code ([listener name] with k=50, ["<listener> is <activity>"] with
            k=25) and, with probability `share_prob`, passes on the most
            important piece of news it retrieved that the listener has not
            heard from it
  plan      at wake-up and at 14:00 each resident retrieves with the focal
            points of plan.py ("<name>'s plan for <date>.", "Important recent
            events for <name>'s life.", k=30 each) and may attend events of the
            day that it retrieved
  retrieve  every retrieval marks the returned memories as accessed (refresh)

All random decisions use counter-based hashing (common random numbers), so
two backends diverge only when a retrieval returns something different.
"""

from __future__ import annotations

import time
import zlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from agentmem import DECAY_PAPER, PROFILES, Weights, make_store, tie_aware_recall
from agentmem.embed import CACHE_DIR, SIFEncoder

from .world import (HOME, LUNCH_PLACES, PLACES, RESIDENTS, STEP_HOURS, STEPS_PER_DAY, VISIT, WORK,
                    News, Resident, date_str, weekday)

MASK = (1 << 64) - 1
KIND_OBS, KIND_CHAT, KIND_NEWS = 0, 1, 2


def _mix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def u01(*keys: int) -> float:
    """Deterministic uniform number in [0, 1) for a tuple of integer keys."""
    h = 0x12345
    for k in keys:
        h = _mix(h ^ (int(k) & MASK))
    return (h >> 11) / float(1 << 53)


def sid(s: str) -> int:
    return zlib.crc32(s.encode())


# ------------------------------------------------------------------ importance

def activity_importance(act: str) -> int:
    if act == "sleeping":
        return 1
    if act.startswith("attending"):
        return 7
    if "chatting" in act:
        return 4
    if any(act == h for h in HOME["morning"] + HOME["evening"]):
        return 2
    return 3


def noisy(base: int, *keys: int) -> float:
    u = u01(*keys)
    delta = -1 if u < 0.25 else (1 if u > 0.75 else 0)
    return min(10, max(1, base + delta)) / 10.0


# ------------------------------------------------------------------ gossip

GOSSIP = [
    ("adopted a puppy", 4, None),
    ("is selling an old bicycle", 3, None),
    ("is learning to play the piano", 3, None),
    ("is looking for a new apartment", 4, None),
    ("got a promotion at work", 6, None),
    ("is moving to another city next month", 7, None),
    ("lost a wallet at the gym", 4, None),
    ("is organizing a picnic at Riverside Park on {wd} at noon", 6, ("Riverside Park", 12.0, 14.0)),
    ("is organizing a charity run at Riverside Park on {wd} morning", 6, ("Riverside Park", 9.0, 11.0)),
    ("is giving a talk at Oak Hill College on {wd} at 6 pm", 5, ("Oak Hill College", 18.0, 19.5)),
    ("is hosting a board game night at the Rose Pub on {wd} at 8 pm", 5, ("Rose Pub", 20.0, 22.0)),
    ("is having a birthday dinner at the Rose Pub on {wd} at 7 pm", 6, ("Rose Pub", 19.0, 21.0)),
]


@dataclass
class SimConfig:
    days: int = 3
    seed: int = 0
    backend: str = "exact"
    backend_kw: dict = field(default_factory=dict)
    weights: Weights = PROFILES["paper"]
    decay: float = DECAY_PAPER
    party_day: int = 1
    party_notice_hours: float = 35.0   # host learns it 35 h before 5 pm (Feb 13 morning in the paper)
    party_appeal: float = 0.45       # calibrated: 5 of 12 invited attended in the paper
    k_conv1: int = 50
    k_conv2: int = 25
    k_plan: int = 30
    att_bandwidth: int = 3
    retention: int = 8
    talk_rate: float = 0.06          # calibrated so that exact retrieval reproduces the paper's
    share_prob: float = 0.5          # diffusion (13 of 25 knew about the party after 2 days)
    cooldown_h: float = 3.0
    gossip_every_h: float = 8.0
    measure_recall: bool = True
    log_every_day: bool = False
    trace_from: float | None = None   # record a frame per step from this hour on (for the demo)


class Agent:
    def __init__(self, idx: int, res: Resident, store):
        self.idx, self.res, self.name = idx, res, res.name
        self.store = store
        self.kind: list[int] = []
        self.nid: list[int] = []
        self.text: list[str] = []
        self.recent: deque[str] = deque(maxlen=0)
        self.today: list[tuple[str, str]] = []
        self.prev: tuple[str, str] | None = None
        self.told: set[tuple[int, int]] = set()      # (listener, news) pairs already covered
        self.hosting: list[int] = []
        self.heard: dict[int, float] = {}            # news -> first time known
        self.planned: dict[int, int] = {}            # news -> day planned


class Town:
    def __init__(self, cfg: SimConfig, encoder: SIFEncoder | None = None):
        self.cfg = cfg
        self.enc = encoder or load_encoder()
        dim = self.enc.dim
        self.agents = [Agent(i, r, make_store(cfg.backend, dim, decay=cfg.decay, seed=i, **cfg.backend_kw))
                       for i, r in enumerate(RESIDENTS)]
        for a in self.agents:
            a.recent = deque(maxlen=cfg.retention)
        self.by_name = {a.name: a for a in self.agents}
        self.news: list[News] = []
        self.last_talk: dict[tuple[int, int], float] = {}
        self.metrics = dict(recall=[], recall_kind=[], recall_T=[], conversations=0, shared=0,
                            retrievals=0)
        self.party = self._add_party()
        self.gossip_i = 0
        self.attend_log: dict[int, set[int]] = {}
        self.present_log: dict[int, set[int]] = {}
        self.plan_hits: dict[int, set[int]] = {}
        self.party_in_context: list[tuple[int, float]] = []   # (agent, T) plan retrievals that contained it
        self.timeline: list[dict] = []
        self.frames: list[list[tuple]] = []
        self.wall = 0.0

    # -------------------------------------------------------------- setup
    def _add_party(self) -> int:
        P = self.cfg.party_day
        n = News("party", f"Maria Lopez is hosting a party at Maple Cafe on {date_str(P)} at 5 pm "
                          f"and everyone is invited", "the party at Maple Cafe", 8, "Maria Lopez",
                 announce=P * 24.0 + 17.0 - self.cfg.party_notice_hours, day=P, place="Maple Cafe",
                 start=17.0, end=19.0, appeal=self.cfg.party_appeal, host_spreads=True)
        self.news.append(n)
        return len(self.news) - 1

    def _new_gossip(self, T: float, day: int) -> None:
        cfg = self.cfg
        i = self.gossip_i
        self.gossip_i += 1
        subj = RESIDENTS[int(u01(cfg.seed, 11, i) * len(RESIDENTS))].name
        pred, imp, ev = GOSSIP[int(u01(cfg.seed, 12, i) * len(GOSSIP))]
        teller = RESIDENTS[int(u01(cfg.seed, 13, i) * len(RESIDENTS))].name if u01(cfg.seed, 14, i) < 0.5 else subj
        ev_day = day + 2 + int(u01(cfg.seed, 15, i) * 3)
        content = f"{subj} {pred.format(wd=weekday(ev_day))}"
        kw = {}
        if ev is not None:
            kw = dict(day=ev_day, place=ev[0], start=ev[1], end=ev[2], appeal=0.35)
        title = f"the news that {subj} {pred.split(' on ')[0].format(wd='')}".strip()
        self.news.append(News(f"g{i}", content, title, imp, teller, announce=T, **kw))
        self._learn(self.by_name[teller], len(self.news) - 1, T, own=True)

    def _learn(self, a: Agent, nid: int, T: float, own: bool) -> None:
        n = self.news[nid]
        imp = noisy(n.importance + (1 if own else 0), a.idx, nid, 77)
        self._remember(a, [n.content], [imp], T, KIND_NEWS, nid)
        a.heard.setdefault(nid, T)
        if own and n.host_spreads:
            a.hosting.append(nid)

    # -------------------------------------------------------------- memory
    def _remember(self, a: Agent, texts: list[str], imps: list[float], T: float,
                  kind: int, nid: int = -1) -> None:
        if not texts:
            return
        E = self.enc.encode(texts)
        a.store.add(E, np.asarray(imps), T)
        a.kind.extend([kind] * len(texts))
        a.nid.extend([nid] * len(texts))
        a.text.extend(texts)

    def _retrieve(self, a: Agent, focal: str, T: float, k: int, tag: int) -> np.ndarray:
        q = self.enc.encode([focal])[0]
        w = self.cfg.weights
        if self.cfg.measure_recall and a.store.n > 0:
            true = a.store.exact(q, T, w)
        ids = a.store.retrieve(q, T, k, w, refresh=True)
        if self.cfg.measure_recall and a.store.n > 0:
            self.metrics["recall"].append(tie_aware_recall(ids, true, k))
            self.metrics["recall_kind"].append(tag)
            self.metrics["recall_T"].append(T)
        self.metrics["retrievals"] += 1
        return ids

    # -------------------------------------------------------------- schedules
    def _day_schedule(self, a: Agent, day: int) -> list[tuple[str, str]]:
        res, cfg = a.res, self.cfg
        rng = np.random.default_rng([cfg.seed, a.idx, day, 4242])
        slots: list[tuple[str, str] | None] = [None] * STEPS_PER_DAY
        home = res.home

        def fill(h0, h1, place, acts, every=1.0):
            s0, s1 = int(round(h0 * 4)), int(round(h1 * 4))
            act = None
            for s in range(max(0, s0), min(STEPS_PER_DAY, s1)):
                if act is None or (s - s0) % max(1, int(every * 4)) == 0:
                    act = acts[int(rng.integers(len(acts)))]
                if slots[s] is None:
                    slots[s] = (place, act)

        def pick(weights: dict[str, float]) -> str:
            names = list(weights)
            p = np.asarray([weights[n] for n in names], dtype=float)
            return names[int(rng.choice(len(names), p=p / p.sum()))]

        wd = day % 7
        works = res.work is not None and wd in res.days
        start, end = res.hours
        wake = 6.0 + 0.25 * int(rng.integers(0, 7))
        if works:
            wake = min(wake, start - 0.75)
        bed = 22.5 + 0.25 * int(rng.integers(0, 6))
        if works and end > 21:   # evening shift (bartender, musician)
            bed = max(bed, min(23.75, end + 0.25))
            fill(start, end, res.work, WORK[res.role])
        fill(0, wake, home, HOME["night"])
        fill(bed, 24, home, HOME["night"])
        if works and start < 14:
            fill(wake, start, home, HOME["morning"], every=0.5)
            workplaces = [res.work]
            if res.role == "developer":
                workplaces = ["Maple Cafe" if rng.random() < 0.6 else "Willow Library"]
            if res.role in ("journalist", "candidate"):
                workplaces = ["Town Hall", "Maple Cafe", "Riverside Park"]
            if start < 12 and end > 13 and res.role not in ("baker",):
                lunch = pick({**LUNCH_PLACES, home: 0.5})
                fill(12, 13, lunch, ["eating lunch"])
            h = start
            while h < end:
                wp = workplaces[int(rng.integers(len(workplaces)))]
                fill(h, min(end, h + 2.0), wp, WORK[res.role])
                h += 2.0
            leisure_start = end
        else:
            fill(wake, min(10.0, wake + 2.5), home, HOME["morning"], every=0.5)
            p1 = pick(res.likes)
            fill(10.0, 12.0, p1, VISIT[PLACES[p1]])
            lunch = pick({**LUNCH_PLACES, home: 1.0})
            fill(12, 13, lunch, ["eating lunch"])
            p2 = pick(res.likes)
            fill(14.0, 16.5, p2, VISIT[PLACES[p2]])
            if rng.random() < 0.4:
                errand = "Harbor Grocery" if rng.random() < 0.7 else "Pine Pharmacy"
                fill(16.5, 17.25, errand, VISIT[PLACES[errand]])
            leisure_start = 17.5
        if leisure_start <= 19.5:
            place = pick(res.likes)
            length = 1.5 + 0.5 * int(rng.integers(0, 4))
            fill(leisure_start, leisure_start + length, place, VISIT[PLACES[place]])
            if rng.random() < 0.25:
                errand = "Harbor Grocery" if rng.random() < 0.7 else "Pine Pharmacy"
                t0 = leisure_start + length
                fill(t0, t0 + 0.5, errand, VISIT[PLACES[errand]])
        for s in range(STEPS_PER_DAY):
            if slots[s] is None:
                fill(s / 4, s / 4 + 0.25, home, HOME["evening"])
        return [x for x in slots]  # type: ignore[misc]

    # -------------------------------------------------------------- planning
    def _plan(self, a: Agent, day: int, T: float, focal_points: list[str]) -> None:
        ids = np.unique(np.concatenate([self._retrieve(a, f, T, self.cfg.k_plan, 1) for f in focal_points]))
        hits = []
        for i in ids.tolist():
            if a.kind[i] != KIND_NEWS:
                continue
            nid = a.nid[i]
            if nid == self.party:
                self.party_in_context.append((a.idx, T))
            n = self.news[nid]
            if n.day == day and nid not in a.planned:
                hits.append(nid)
        for nid in a.hosting:
            if self.news[nid].day == day and nid not in a.planned:
                hits.append(nid)
        if self.party in hits:
            self.plan_hits.setdefault(day, set()).add(a.idx)
        for nid in sorted(set(hits), key=lambda j: -self.news[j].importance):
            n = self.news[nid]
            if (T % 24.0) >= n.end:
                continue
            appeal = 1.0 if nid in a.hosting else n.appeal
            if u01(self.cfg.seed, a.idx, nid, 99) < appeal:
                s0, s1 = int(n.start * 4), int(n.end * 4)
                cur = int((T % 24.0) * 4)
                for s in range(max(s0, cur + 1), s1):
                    a.today[s] = (n.place, f"attending {n.title}")
                a.planned[nid] = day

    # -------------------------------------------------------------- conversations
    def _news_to_tell(self, sp: Agent, li: Agent, T: float) -> int | None:
        day = int(T // 24)
        for nid in sp.hosting:
            n = self.news[nid]
            if (li.idx, nid) not in sp.told and (n.day is None or n.day >= day):
                return nid
        listener_act = li.today[int((T % 24.0) * 4)][1]
        ids1 = self._retrieve(sp, li.name, T, self.cfg.k_conv1, 0)
        ids2 = self._retrieve(sp, f"{li.name} is {listener_act}", T, self.cfg.k_conv2, 0)
        if u01(self.cfg.seed, sp.idx, li.idx, int(T * 4), 31) >= self.cfg.share_prob:
            return None
        best, best_key = None, None
        for i in np.union1d(ids1, ids2).tolist():
            if sp.kind[i] != KIND_NEWS:
                continue
            nid = sp.nid[i]
            n = self.news[nid]
            if (li.idx, nid) in sp.told:
                continue
            if n.day is not None and (n.day < day or (n.day == day and (T % 24.0) >= n.end)):
                continue
            key = (n.importance, -nid)
            if best_key is None or key > best_key:
                best, best_key = nid, key
        return best

    def _converse(self, a: Agent, b: Agent, T: float) -> None:
        self.metrics["conversations"] += 1
        shared = False
        for sp, li in ((a, b), (b, a)):
            nid = self._news_to_tell(sp, li, T)
            if nid is None:
                continue
            n = self.news[nid]
            self._remember(li, [f"{sp.name} told me that {n.content}"],
                           [noisy(n.importance, li.idx, nid, sid(sp.name))], T, KIND_NEWS, nid)
            self._remember(sp, [f"{sp.name} told {li.name} about {n.title}"],
                           [noisy(3, sp.idx, nid, li.idx)], T, KIND_CHAT)
            sp.told.add((li.idx, nid))
            li.told.add((sp.idx, nid))
            li.heard.setdefault(nid, T)
            self.metrics["shared"] += 1
            shared = True
        if not shared:
            common = sorted(set(a.res.interests) & set(b.res.interests)) or list(a.res.interests)
            topic = common[int(u01(self.cfg.seed, a.idx, b.idx, int(T * 4)) * len(common))]
            text = f"{a.name} and {b.name} chatted about {topic}"
            self._remember(a, [text], [noisy(3, a.idx, b.idx, int(T * 4))], T, KIND_CHAT)
            self._remember(b, [text], [noisy(3, b.idx, a.idx, int(T * 4))], T, KIND_CHAT)

    # -------------------------------------------------------------- main loop
    def run(self) -> dict:
        cfg = self.cfg
        t_start = time.perf_counter()
        next_gossip = cfg.gossip_every_h
        for day in range(cfg.days):
            for a in self.agents:
                a.today = self._day_schedule(a, day)
            for step in range(STEPS_PER_DAY):
                T = day * 24.0 + step * STEP_HOURS
                # news appear
                for nid, n in enumerate(self.news):
                    if n.announce <= T < n.announce + STEP_HOURS and n.teller and nid not in self.by_name[n.teller].heard:
                        self._learn(self.by_name[n.teller], nid, T, own=True)
                while T >= next_gossip:
                    self._new_gossip(T, day)
                    next_gossip += cfg.gossip_every_h
                # planning at wake-up and at 14:00
                for a in self.agents:
                    cur, prev = a.today[step], a.today[step - 1] if step else ("", "sleeping")
                    if prev[1] == "sleeping" and cur[1] != "sleeping" and step < 48:
                        self._plan(a, day, T, [f"{a.name}'s plan for {date_str(day)}.",
                                               f"Important recent events for {a.name}'s life."])
                    elif step == 56:
                        self._plan(a, day, T, [f"{a.name}'s plan for this evening, {date_str(day)}."])
                # perception
                groups: dict[str, list[Agent]] = {}
                for a in self.agents:
                    groups.setdefault(a.today[step][0], []).append(a)
                for place, members in groups.items():
                    for a in members:
                        texts, imps = [], []
                        now = a.today[step]
                        if now != a.prev:
                            t = f"{a.name} is {now[1]} at {place}"
                            if t not in a.recent:
                                texts.append(t)
                                imps.append(noisy(activity_importance(now[1]), a.idx, day, step, 1))
                                a.recent.append(t)
                            a.prev = now
                        others = [b for b in members if b is not a]
                        others.sort(key=lambda b: u01(cfg.seed, day, step, a.idx, b.idx, 5))
                        for b in others[: cfg.att_bandwidth]:
                            t = f"{b.name} is {b.today[step][1]} at {place}"
                            if t in a.recent:
                                continue
                            texts.append(t)
                            imps.append(noisy(max(1, activity_importance(b.today[step][1]) - 1),
                                              a.idx, b.idx, day, step))
                            a.recent.append(t)
                        self._remember(a, texts, imps, T, KIND_OBS)
                # attendance at events
                for nid, n in enumerate(self.news):
                    if n.day == day and n.start * 4 <= step < n.end * 4:
                        for a in groups.get(n.place, []):
                            self.present_log.setdefault(nid, set()).add(a.idx)
                            if a.today[step][1].startswith("attending"):
                                self.attend_log.setdefault(nid, set()).add(a.idx)
                # conversations
                for place, members in groups.items():
                    awake = [a for a in members if a.today[step][1] != "sleeping"]
                    if len(awake) < 2:
                        continue
                    pairs = [(x, y) for i, x in enumerate(awake) for y in awake[i + 1:]]
                    pairs.sort(key=lambda p: u01(cfg.seed, day, step, p[0].idx, p[1].idx, 6))
                    busy: set[int] = set()
                    for x, y in pairs:
                        if x.idx in busy or y.idx in busy:
                            continue
                        key = (min(x.idx, y.idx), max(x.idx, y.idx))
                        if T - self.last_talk.get(key, -1e9) < cfg.cooldown_h:
                            continue
                        p = cfg.talk_rate * x.res.sociability * y.res.sociability
                        if u01(cfg.seed, day, step, key[0], key[1], 7) < p:
                            self._converse(x, y, T)
                            self.last_talk[key] = T
                            busy.update((x.idx, y.idx))
                if step % 4 == 0:
                    self.timeline.append(dict(T=T, aware=sum(1 for a in self.agents if self.party in a.heard)))
                if cfg.trace_from is not None and T >= cfg.trace_from:
                    self.frames.append([(a.today[step][0], int(self.party in a.heard),
                                         int(a.today[step][1].startswith("attending the party")),
                                         a.today[step][1]) for a in self.agents])
            if cfg.log_every_day:
                mem = np.mean([a.store.n for a in self.agents])
                rec = np.mean(self.metrics["recall"]) if self.metrics["recall"] else float("nan")
                print(f"  day {day}: mean memories {mem:.0f}, retrieval recall {rec:.3f}, "
                      f"{time.perf_counter() - t_start:.0f}s", flush=True)
        self.wall = time.perf_counter() - t_start
        return self.summary()

    def summary(self) -> dict:
        P = self.party
        party_T = self.cfg.party_day * 24.0 + 17.0
        rec = np.asarray(self.metrics["recall"])
        kinds = np.asarray(self.metrics["recall_kind"])
        st = [a.store.stats for a in self.agents]
        return dict(
            backend=self.agents[0].store.name,
            seed=self.cfg.seed,
            days=self.cfg.days,
            party_day=self.cfg.party_day,
            aware=sum(1 for a in self.agents if P in a.heard),
            aware_at_party=sum(1 for a in self.agents if a.heard.get(P, 1e18) < party_T),
            planned=sum(1 for a in self.agents if P in a.planned),
            attended=len(self.attend_log.get(P, set())),
            present=len(self.present_log.get(P, set())),
            recalled_at_plan=len(self.plan_hits.get(self.cfg.party_day, set())),
            mean_memories=float(np.mean([a.store.n for a in self.agents])),
            conversations=self.metrics["conversations"],
            shared=self.metrics["shared"],
            retrievals=self.metrics["retrievals"],
            recall=float(rec.mean()) if len(rec) else float("nan"),
            recall_conv=float(rec[kinds == 0].mean()) if np.any(kinds == 0) else float("nan"),
            recall_plan=float(rec[kinds == 1].mean()) if np.any(kinds == 1) else float("nan"),
            search_ms=1000 * sum(s["search_s"] for s in st) / max(1, sum(s["queries"] for s in st)),
            insert_ms=1000 * sum(s["insert_s"] for s in st) / max(1, sum(s["inserted"] for s in st)),
            refresh_ms=1000 * sum(s["refresh_s"] for s in st) / max(1, sum(s["queries"] for s in st)),
            wall_s=self.wall,
        )


def load_encoder() -> SIFEncoder:
    """SIF encoder with the common component fitted once on town vocabulary (cached)."""
    enc = SIFEncoder()
    pc_path = CACHE_DIR / "sif_pc_town.npy"
    if pc_path.exists():
        enc.pc = np.load(pc_path)
        return enc
    texts = []
    for r in RESIDENTS:
        for place, kind in PLACES.items():
            for act in VISIT[kind] + (WORK.get(r.role, []) if r.role else []):
                texts.append(f"{r.name} is {act} at {place}")
        for acts in HOME.values():
            for act in acts:
                texts.append(f"{r.name} is {act} at {r.home}")
        for pred, _, _ in GOSSIP:
            texts.append(f"{r.name} {pred.format(wd='Friday')}")
    enc.fit(texts)
    np.save(pc_path, enc.pc)
    return enc
