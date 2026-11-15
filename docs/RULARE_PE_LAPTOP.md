# Rularea pe laptop: instalare, tur și validarea cu un LLM local

Documentul are două părți. Prima e promptul pe care îl lipești în Claude Code după ce ai clonat proiectul. A doua sunt aceiași pași scriși pentru om, ca să poți urmări ce face Claude Code sau să îi faci tu de mână.

Scopul rulării de pe laptop e unul singur: **validarea cu un model de limbaj local**. E singurul experiment din proiect care nu s-a putut rula în cloud, pentru că are nevoie de placă video. Restul rezultatelor sunt deja în `results/` și nu trebuie refăcute.

## De ce contează validarea

Agenții din oraș urmează reguli. Întrebarea la care trebuie să te aștepți la prezentare e „de unde știi că un LLM adevărat nu ar compensa o amintire lipsă?”. Validarea răspunde direct: aceleași orașe, aceleași seed-uri, dar din momentul în care Maria află de petrecere două decizii le ia modelul (ce veste spune într-o conversație și la ce evenimente merge azi). Se compară trei politici de regăsire: scorul complet din articol, overfetch + reordonare ca în LangChain și doar relevanța.

Orice rezultat e bun de raportat:
- dacă ordinea politicilor e aceeași ca la reguli, mesajul al treilea al proiectului („regăsirea schimbă comportamentul”) nu mai depinde de reguli
- dacă modelul șterge diferența, o spui ca limită, cu cifre. E tot un rezultat și valorează mai mult decât o afirmație neverificată

De ce 10 seed-uri: testul e Wilcoxon pe perechi, cu corecție Holm pentru 6 comparații. Sub 8 seed-uri nu poate ieși semnificativ nici dacă toate perechile merg în aceeași direcție. Cu 10 are loc.

## Partea 1: promptul pentru Claude Code

Clonezi proiectul, deschizi Claude Code în folderul lui și lipești tot blocul de mai jos.

```
git clone https://github.com/bogdancolceriu/agentmem-hnsw
cd agentmem-hnsw
```

```text
Ești în proiectul meu de seminar agentmem-hnsw (o extensie a articolului HNSW către memoria
agenților generativi). Citește întâi CLAUDE.md, README.md, docs/PLAN_REVIZIE.md și
docs/RULARE_PE_LAPTOP.md. Sunt pe Windows, cu o placă video NVIDIA RTX 4050 (6 GB).
Vorbește cu mine în română, simplu, fără liniuțe lungi și fără punct și virgulă.

Ce vreau, în ordinea asta. Oprește-te și spune-mi dacă un pas nu iese, nu trece peste el.

FAZA 1. Instalare și verificare
1. Verifică Python 3.12 (py -3.12 --version) și Git. Dacă lipsește Python 3.12, spune-mi ce să
   instalez, nu instala altă versiune în locul lui.
2. Fă un mediu virtual .venv în proiect și instalează requirements.txt. Dacă hnswlib nu se
   compilează pe Windows, urmează varianta din docs/ghid_aparare.md, secțiunea 9, pasul 3.
3. Rulează python -m pytest -q tests/ . Trebuie să treacă 45 de teste. Dacă pică vreunul,
   arată-mi eroarea și oprește-te.
4. Rulează python demo/tour.py și explică-mi pe scurt, pas cu pas, ce arată fiecare parte și în
   ce fișier e codul. Vreau să pot explica și eu.

FAZA 2. Pregătirea modelului
5. Verifică dacă Ollama e instalat (ollama --version). Dacă nu, spune-mi să îl instalez de pe
   https://ollama.com/download și așteaptă să confirm. Nu descărca tu installere.
6. Verifică dacă modelul qwen2.5:7b e descărcat (ollama list). Dacă nu, cere-mi voie și apoi
   rulează ollama pull qwen2.5:7b (cam 4,7 GB).
7. Testează lanțul fără model, cu ieșirea într-un folder temporar din afara proiectului
   (%TEMP% în cmd, $env:TEMP în PowerShell):
   python -m experiments.llm_validation --mock --seeds 1 --out %TEMP%\llm_test.csv
8. Rulare pilot cu modelul, tot în afara proiectului:
   python -m experiments.llm_validation --seeds 1 --policies exact --out %TEMP%\llm_pilot.csv
   În timpul ei verifică cu ollama ps că scrie 100% GPU. Spune-mi câte minute a durat și
   estimează cât durează 10 seed-uri cu 3 politici. Dacă modelul rulează parțial pe procesor
   sau un apel durează peste 10 secunde, oprește-te și propune-mi variante (driver NVIDIA,
   închis alte programe, sau modelul qwen2.5:3b), nu schimba tu modelul.

FAZA 3. Experimentul
9. După ce confirm, pornește rularea reală în fundal:
   python -m experiments.llm_validation --seeds 10
   Scrie în results/llm_validation.csv și se poate relua: dacă se întrerupe, aceeași comandă
   sare peste rulările terminate. Folosește un singur model pentru toate rândurile.
   Poate dura mai multe ore. Verifică din când în când și anunță-mă când e gata.

FAZA 4. Rezultatele în raport
10. Rulează python -m experiments.tables . Apoi git diff report/generated/numbers.tex trebuie
    să arate DOAR linii adăugate, toate cu macro-uri care încep cu Llm, plus fișierul nou
    report/generated/tab_llm.tex. Dacă se schimbă orice alt număr, oprește-te și arată-mi.
11. Arată-mi tabelul și spune-mi direct ce arată: ordinea celor trei politici e aceeași cu
    modelul ca la reguli? Cât de mare e scăderea la „doar relevanța”? Ce p ajustat Holm are?
    Ce nu se poate spune pe baza a 10 seed-uri? Nu înfrumuseța.
12. În report/llm_section.tex înlocuiește comentariul % INTERPRETATION cu două sau trei
    propoziții în engleză britanică, în stilul restului raportului (fără liniuțe lungi, fără
    punct și virgulă, fără virgulă înainte de „and”), folosind macro-urile Llm, nu cifre scrise
    de mână. Nu afirma mai mult decât arată tabelul.
13. Dacă am LaTeX instalat (latexmk --version), compilează:
    latexmk -pdf main.tex   (din folderul report)
    Verifică să nu fie referințe nerezolvate. Dacă nu am LaTeX, sari peste și spune-mi.
14. În docs/ghid_aparare.md, la întrebarea „De ce n-ai folosit un LLM real pentru agenți?”,
    înlocuiește paranteza „(Dacă ai rulat-o...)” cu rezultatul real, într-o propoziție sau două.
    În docs/PLAN_REVIZIE.md bifează punctul C în jurnal.

FAZA 5. Încheiere
15. Arată-mi git status și git diff --stat. După ce confirm, fă un singur commit cu mesajul
    „LLM validation: results on 10 seeds” și dă push.
16. La final scrie-mi un rezumat de cel mult 10 rânduri pe care să îl lipesc în conversația în
    care lucrez la prezentare: modelul, numărul de seed-uri, tabelul cu cele trei politici
    (reguli și model) și concluzia într-o frază.

Reguli pentru toată sesiunea:
- Nu rerula experimentele mari (exp_hnsw, exp_static, exp_town, exp_town_curve). Rezultatele lor
  sunt în results/ și trebuie să rămână neschimbate.
- Niciun număr din raport nu se scrie de mână. Toate vin din experiments/tables.py.
- Nu șterge și nu rescrie fișiere din results/, figures/ sau report/ în afara celor de mai sus.
- Nu face commit și nu da push fără să te confirm eu.
- Dacă nu ești sigur de ceva, întreabă-mă.
```

Dacă vrei doar să înțelegi proiectul, fără validare, lipește doar fazele 1 și regulile, apoi pune-i întrebări despre cod („unde se construiește vectorul pliat?”, „de ce are nevoie memoria de segmente?”).

## Partea 2: aceiași pași, de mână

### Instalare

1. Python 3.12 de pe python.org (bifează „Add python.exe to PATH”) și Git.
2. Într-un Command Prompt:
   ```
   git clone https://github.com/bogdancolceriu/agentmem-hnsw
   cd agentmem-hnsw
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m pytest -q tests/
   python demo/tour.py
   ```
   Trebuie să vezi 45 de teste trecute. Dacă `hnswlib` nu se instalează, vezi `docs/ghid_aparare.md`, secțiunea 9. Validarea cu LLM merge și fără el, pentru că folosește doar scanări exacte.

### Ollama și modelul

3. Instalează Ollama de pe https://ollama.com/download. După instalare rulează singur în fundal.
4. Descarcă modelul (cam 4,7 GB) și verifică-l:
   ```
   ollama pull qwen2.5:7b
   ollama run qwen2.5:7b "Salut"
   ```

### Experimentul

5. Test fără model, apoi pilot cu un singur seed:
   ```
   python -m experiments.llm_validation --mock --seeds 1 --out %TEMP%\llm_test.csv
   python -m experiments.llm_validation --seeds 1 --policies exact --out %TEMP%\llm_pilot.csv
   ```
   În alt terminal, `ollama ps` trebuie să arate `100% GPU`. La prima rulare se descarcă vectorii GloVe (cam 130 MB).
6. Rularea reală:
   ```
   python -m experiments.llm_validation --seeds 10
   ```
   Pentru fiecare seed și politică se rulează orașul de 32 de zile o dată cu reguli și o dată cu modelul, deci 60 de rulări. O rulare cu modelul face între 130 și 225 de apeluri. Durata depinde de placa video, o afli din pilot. Poți opri oricând cu Ctrl+C și relua cu aceeași comandă.

### Rezultatele în raport

7. `python -m experiments.tables` scrie `report/generated/tab_llm.tex` și macro-urile `Llm...` în `report/generated/numbers.tex`. Din acel moment raportul include singur subsecțiunea „A check with a language model in the loop” (fișierul `report/llm_section.tex`) și schimbă fraza corespunzătoare de la limite.
8. În `report/llm_section.tex` înlocuiești comentariul `% INTERPRETATION` cu două sau trei propoziții despre ce arată tabelul.
9. Compilezi raportul (`cd report`, apoi `latexmk -pdf main.tex`), dacă ai LaTeX. Dacă nu ai, trimite-mi `results/llm_validation.csv` și îl compilez eu.
10. Commit și push. Apoi aduci rezumatul în conversația cu prezentarea, ca să actualizez slide-ul de limite și ghidul.

### Dacă ceva nu merge

- **„Ollama did not answer”**: Ollama nu rulează. Pornește aplicația din meniul Start și reia comanda.
- **Foarte încet (peste 10 secunde la un apel)**: modelul rulează pe procesor. Verifică `ollama ps`. Închide programele care ocupă memoria plăcii video (jocuri, browser cu multe taburi) și actualizează driverul NVIDIA. Ca ultimă variantă, `--model qwen2.5:3b`, dar atunci toate cele 10 seed-uri se rulează cu el.
- **Ai rulat din greșeală cu două modele**: tabelul îl folosește pe cel cu cele mai multe rânduri. Mai curat e să ștergi din `results/llm_validation.csv` rândurile celuilalt model.
- **S-a întrerupt curentul sau ai închis laptopul**: reiei aceeași comandă. Rulările terminate nu se refac.
