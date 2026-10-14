# Validarea cu un LLM local (opțională, pe laptopul tău)

Scopul: să vedem dacă efectul din simulare apare și când deciziile agenților le ia un model de limbaj, nu o regulă. Rulează pe placa ta video (RTX 4050, 6 GB) prin Ollama. Durează în jur de o oră, iar tu trebuie doar să pornești comenzile de mai jos.

## Ce face scriptul

Orașul trăiește 30 de zile cu agenți pe reguli, apoi, din momentul în care Maria Lopez află de petrecere, două decizii le ia modelul:

- **în conversații:** vorbitorul vede veștile din amintirile pe care i le-a dat metoda de regăsire și modelul alege ce veste (dacă vreuna) îi spune celuilalt;
- **la planificare:** agentul vede toate amintirile regăsite pentru planul zilei și modelul scrie la ce evenimente merge azi.

Se rulează trei metode de regăsire (scorul din articol, overfetch + re-rank ca în LangChain, doar relevanță), câte 3 seed-uri. Pentru fiecare rulare se notează câți agenți au aflat de petrecere, câți și-au amintit-o la planificare și câți au venit.

## Pașii (Windows)

1. **Instalează Ollama** de pe https://ollama.com/download (installer-ul pentru Windows). După instalare rulează singur în fundal.
2. **Descarcă modelul.** Deschide PowerShell și scrie:
   ```
   ollama pull qwen2.5:7b
   ```
   Are cam 4,7 GB. Dacă vrei să verifici că merge: `ollama run qwen2.5:7b "Salut"` (ieși cu Ctrl+D).
3. **Ia proiectul** (arhiva pe care ți-am trimis-o sau repo-ul de pe GitHub) și dezarhivează-l, de exemplu în `C:\proiecte\agentmem-hnsw`.
4. **Instalează numpy** (singura dependență pentru partea asta; ai nevoie de Python 3.10 sau mai nou):
   ```
   cd C:\proiecte\agentmem-hnsw
   py -m pip install numpy
   ```
5. **Pornește experimentul:**
   ```
   py -m experiments.llm_validation --model qwen2.5:7b
   ```
   La prima rulare descarcă singur vectorii GloVe (~130 MB). Pe ecran apare câte o linie la fiecare rulare terminată (9 în total).
6. **Trimite-mi** fișierele `results\llm_validation.csv` și `results\llm_validation_log.jsonl`. Le analizez și adaug rezultatul în raport și în prezentare.

## Dacă ceva nu merge

- `ConnectionRefusedError` sau `urlopen error`: Ollama nu rulează. Pornește aplicația Ollama din meniul Start și încearcă din nou.
- Merge foarte încet (peste 10 secunde la un apel): modelul rulează pe procesor, nu pe placa video. Verifică în `ollama ps` coloana `PROCESSOR`; ar trebui să scrie `100% GPU`. Dacă nu, actualizează driverul NVIDIA.
- Vrei doar să vezi că totul e legat corect, fără model: `py -m experiments.llm_validation --mock --seeds 1`.
