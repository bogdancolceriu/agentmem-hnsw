# Plan de revizie (noiembrie 2026)

Scopul: proiectul să spună o singură poveste, ușor de prezentat și de apărat, fără să se piardă
nimic din ce e făcut. Nimic nu se șterge. Rezultatele secundare trec în anexe și în slide-uri de
rezervă.

Starea fiecărui pas e la final, în „Jurnal”. Dacă o sesiune se întrerupe, următoarea continuă de
la primul pas nebifat.

## De ce

Analiza de la 15 noiembrie a găsit patru slăbiciuni:

1. Agenții din simulare merg pe reguli, fără niciun LLM. Validarea cu LLM local există ca script
   (`experiments/llm_validation.py`), dar nu a fost rulată.
2. Rezumatul conduce cu cifra cea mai favorabilă (0,975 față de 0,006, pe date sintetice). Pe
   date realiste diferența e mai mică: 0,96 față de 0,28 pe amintirile din oraș și 1,00 față de
   0,52 cu clasele LangChain.
3. La memoria unui singur agent (câteva mii de amintiri) scanarea exactă e mai rapidă decât
   indexul. Trebuie spus direct, nu doar la limite.
4. Proiectul e prea întins: cinci contribuții, 21 de pagini, 25 de slide-uri.

## Firul roșu: trei mesaje

1. **Articolul.** HNSW, rescris după pseudocod, se comportă cum spune articolul.
2. **Extensia.** Indexul poate căuta direct scorul agenților (recență + importanță + relevanță),
   inclusiv cu alte curbe de uitare, în loc de „ia 100 după similaritate și reordonează”.
3. **De ce contează.** Într-un oraș simulat, felul în care agenții își caută și își uită
   amintirile schimbă ce fac.

Tot restul (ablații, comparația cu IVF, cazul în care domină relevanța, eroarea din codul public
Generative Agents) e material de rezervă: anexe în raport, slide-uri după „Întrebări?”.

## Pași

### A. Raportul (`report/`)
- A1. Rezumat nou, pe cele trei mesaje, cu cifrele realiste primele și cu „agenți pe reguli”
  spus explicit.
- A2. Introducere: trei contribuții în loc de cinci, plus o frază despre anexe.
- A3. Secțiunea 5 (benchmark): rezultatele pe amintirile din oraș primele, testul la un milion
  prezentat ca test de stres. O frază clară: la câteva mii de amintiri scanezi exact, indexul
  contează de la zeci de mii în sus.
- A4. Mutate în Anexa A: tabelul cu estimarea amplitudinii, variația dintre construcții,
  ablațiile, cazul în care domină relevanța.
- A5. Mutată în Anexa B: verificarea codului public Generative Agents. Figura cu recall-ul în
  timp rămâne în secțiunea despre oraș, ca figură separată.
- A6. Concluzie scurtată pe cele trei mesaje.
- A7. Loc pregătit pentru validarea cu LLM: `report/llm_section.tex` intră automat în raport
  când există `report/generated/tab_llm.tex` (făcut de `experiments/tables.py` din
  `results/llm_validation.csv`).
- A8. Compilare, verificat referințele, numerele și paginile.

### B. README și ghid
- B1. README: rezumat pe cele trei mesaje, cifrele realiste primele.
- B2. Ghid (`docs/ghid_aparare.md`): la început „Firul roșu” și „Ce prezinți și ce e rezervă”.
  Trimiterile la secțiunile raportului aduse la zi. Răspuns pregătit la „de ce nu scanezi pur și
  simplu?”.

### C. Validarea cu LLM, pe laptop
- C1. `experiments/llm_validation.py`: reluare după întrerupere, reîncercare la erori de rețea.
- C2. `experiments/tables.py`: tabel și cifre din `results/llm_validation.csv`.
- C3. `docs/RULARE_PE_LAPTOP.md`: pașii pentru Windows și promptul pentru Claude Code.

### D. Prezentarea
- D1. Slide-urile principale urmează cele trei mesaje (cam 16). Restul trec după „Întrebări?”,
  ca rezervă.
- D2. Slide-ul cu un milion de amintiri arată întâi cifrele realiste.
- D3. Numerele de pagină refăcute.

### E. Documentul cu raportul (copia din Claude)
- E1. Adus la zi după noua structură a raportului.

### F. Final
- F1. Toate testele, commit, push.

## Jurnal

- [x] A. Raportul (15.11: rezumat, introducere, secțiunea 5, anexele A și B, concluzie, locul pentru validarea cu LLM)
- [ ] B. README și ghid
- [ ] C. Validarea cu LLM, pregătită pentru laptop
- [ ] D. Prezentarea
- [ ] E. Documentul
- [ ] F. Final
