# Ghid de pregătire: prezentare și întrebări

Ghidul ăsta te pregătește pentru prezentarea din 17.12 și pentru întrebările de după. Citește-l o dată de la cap la coadă, apoi revino la secțiunile unde nu ai putea răspunde fără să te uiți. Cifrele exacte sunt în raport (`report/main.pdf`), aici sunt doar cele de care chiar ai nevoie. Secțiunea 9 spune cum s-a lucrat și cum rulezi tu proiectul, inclusiv un tur ghidat de două minute (`demo/tour.py`).

---

## 0. De la zero: noțiunile pe care se sprijină tot proiectul

Restul ghidului presupune noțiunile de aici. Dacă le știi pe acestea, secțiunile 2–5 se citesc ușor.

### Ce e obligatoriu și ce e extensie

- **Prezentarea articolului 11 (obligatorie):** problema căutării de vecini, NSW, HNSW cu nivelurile lui, căutarea, inserarea, euristica de vecini, parametrii, rezultatele din articol, impactul și limitele. Slide-urile 1–7 și secțiunea 2 din ghid. Atât ar fi prezentat oricine.
- **Verificarea articolului (în plus, dar încă despre articolul 11):** HNSW rescris de tine după pseudocod, comparat cu hnswlib, plus studiul parametrilor refăcut. Slide-urile 8–11 și secțiunea 3.
- **Extensia propriu-zisă:** legătura cu articolul 12 (Generative Agents), scorul ca produs scalar, memoria cu segmente, experimentele pe un milion de amintiri, orașul simulat, a doua extensie (curbele de uitare din psihologie), bug-ul din codul public și retriever-ul pentru LangChain. Slide-urile 12–23 și secțiunile 4–6. Curbele de uitare sunt slide-urile 20–21 și secțiunea 6.
- Slide-urile 24–25 sunt limitele și concluziile întregului proiect.

### Partea 1: căutarea de vecini (lumea articolului 11)

- **Vector, embedding.** Un text (sau o imagine) transformat într-o listă de numere, de exemplu 100 sau 384, astfel încât texte cu sens apropiat să dea vectori apropiați. Programul care face transformarea se numește encoder sau model de embedding. **GloVe** (Stanford, 2014) dă câte un vector de 100 de numere pentru fiecare cuvânt. Vectorul unei propoziții îl obții ca medie ponderată a cuvintelor (metoda SIF). Numărul de numere e **dimensiunea** `d`.
- **Cât de apropiați sunt doi vectori.** Trei măsuri:
  - **distanța euclidiană (L2):** lungimea segmentului dintre ei.
  - **similaritatea cosinus:** cât de mic e unghiul dintre ei, între −1 și 1.
  - **produsul scalar (inner product):** `x·y = x₁y₁ + x₂y₂ + …`. Dacă vectorii au lungimea 1, produsul scalar e chiar cosinusul, iar ordinea după cosinus e aceeași cu ordinea după L2.
- **k-NN (k nearest neighbours).** Dintr-o mulțime de `n` vectori, îi vrei pe cei `k` mai apropiați de o interogare `q`. Varianta exactă (**brute force**, scanare liniară) calculează distanța până la toți, deci costul crește cu `n·d`.
- **Blestemul dimensionalității.** Structurile clasice (arbori kd) taie spațiul în bucăți și funcționează bine în 2–3 dimensiuni. Peste câteva zeci de dimensiuni ajung să verifice aproape tot, deci nu mai câștigi nimic față de scanare.
- **ANN (approximate nearest neighbours).** Accepți să ratezi uneori un vecin în schimbul unei viteze de sute de ori mai mari. Calitatea se măsoară prin **recall@k**: dacă adevărații 10 vecini sunt A…J și tu ai găsit 9 dintre ei, recall@10 = 0,9.
- **Familii de indexuri ANN:** hashing (LSH), cuantizare și clustere (**IVF**: împarți vectorii în clustere și cauți doar în cele mai apropiate câteva) și **grafuri** (NSW, HNSW). HNSW face parte din a treia familie.
- **Graf de proximitate și căutare lacomă.** Fiecare vector e un nod legat de câțiva vecini apropiați. Ca să cauți, pleci dintr-un nod și te muți mereu la vecinul cel mai apropiat de `q`, până nu mai ai unde să te apropii.
- **Small world, navigabil.** Ca la „șase grade de separare”: dacă pe lângă legăturile scurte există și câteva lungi, orice nod e la puțini pași de oricare altul. Graful e **navigabil** dacă și căutarea lacomă, care vede doar vecinii nodului curent, găsește drumul scurt.
- **Skip list.** O listă ordonată cu mai multe niveluri: sus puține elemente, cu salturi mari, jos toate, cu pași mici. Cauți sus, cobori când ai trecut de țintă. HNSW e exact ideea asta mutată pe grafuri: nivelurile de sus au puține noduri și legături lungi.
- **Fascicul (beam), `ef`.** În loc să ții un singur „cel mai bun nod curent”, ții o listă cu cei mai buni `ef` candidați. Un `ef` mai mare înseamnă recall mai mare și căutare mai lentă. E butonul cu care alegi compromisul.
- **Hub.** Un nod cu foarte multe legături. În dimensiuni mari apar singure și fac o parte din treaba nivelurilor de sus (vezi „Down with the Hierarchy”, secțiunea 3).
- **Evaluări de distanță.** Câte distanțe calculează o căutare. E costul pe care îl compară articolul, pentru că nu depinde de calculator.
- **Bază de date vectorială.** Un sistem care stochează vectori și răspunde la căutări de vecini (pgvector, Qdrant, Milvus, Weaviate). Aproape toate folosesc HNSW. Folosirea tipică e **RAG**: cauți textele relevante și i le dai unui LLM ca context.
- **MIPS (maximum inner product search).** Căutarea vectorilor cu cel mai mare produs scalar cu `q`, nu a celor mai apropiați. Diferența contează când vectorii nu au toți lungimea 1: produsul scalar nu e o distanță adevărată, deci garanțiile teoretice ale HNSW nu se aplică direct. În practică HNSW merge bine și așa (Morozov și Babenko, 2018), iar hnswlib are metrica `ip` pentru asta. Extensia ta se bazează pe MIPS.

### Partea 2: agenții generativi (lumea articolului 12)

- **Agent cu LLM.** Un program care folosește un model de limbaj ca să decidă ce face: percepe ce se întâmplă, își amintește, planifică și acționează, în buclă.
- **Generative Agents** (Park et al., 2023). 25 de personaje într-un orășel ca în The Sims, fiecare condus de ChatGPT. Comportamentul interesant apare singur: un agent decide să dea o petrecere de Valentine's Day, vestea circulă prin conversații, iar o parte din cei invitați chiar vin. Articolul a numărat câți au aflat (13 din 25) și câți invitați au venit (5 din 12).
- **Memory stream.** Memoria unui agent: lista tuturor observațiilor lui, scrise ca text („Maria discută cu Tom la cafenea”). Fiecare amintire are momentul creării, momentul ultimului acces, importanța și embedding-ul ei.
- **Regăsirea (retrieval).** Când agentul trebuie să decidă ceva (ce vorbește, ce face azi), formulează o întrebare către memorie (în cod se numește **focal point**) și primește top-k amintiri după scorul `recență + importanță + relevanță`. Detaliile scorului sunt în secțiunea 4.
- **Recența exponențială.** `0,995` la puterea orelor de la ultimul acces: după o zi rămâne 0,89, după o săptămână 0,43, după o lună 0,03. Fiecare oră înmulțește cu același factor. Pe asta se sprijină toată extensia.
- **Refresh la acces.** O amintire regăsită primește ca „ultim acces” momentul de acum, deci redevine recentă. Amintirile folosite des rămân „vii”.
- **Normalizarea min-max.** Fiecare dintre cei trei termeni e rescalat la [0, 1] peste toate amintirile: cel mai mic devine 0, cel mai mare 1. Așa niciun termen nu domină doar pentru că are altă unitate de măsură.
- **Reflecție și planificare.** Agentul scrie periodic concluzii despre propriile amintiri (care intră tot în memorie) și își face planul zilei pe baza a ce regăsește. Și ele trec prin aceeași regăsire.
- **LangChain și overfetch.** LangChain e cea mai folosită bibliotecă pentru aplicații cu LLM. Clasa ei `TimeWeightedVectorStoreRetriever` implementează memoria în stilul Generative Agents așa: cere bazei vectoriale cele mai asemănătoare 100 de amintiri (**overfetch**), adaugă ultimele create și le ordonează după scorul complet (**re-rank**). Ce nu intră printre cei 100 nu e evaluat niciodată.

### Partea 3: noțiunile extensiei

- **Ideea, într-o frază.** Scorul agentului poate fi scris exact ca produs scalar între vectorul amintirii și vectorul întrebării, fiecare cu două coordonate în plus. Atunci top-k după scor e o căutare MIPS, pe care un singur index HNSW o face direct, fără overfetch.
- **De ce merge.** Exponențiala se desparte: `0,995^(acum − t) = 0,995^(acum − τ) · 0,995^(τ − t)`. Primul factor depinde doar de momentul întrebării, al doilea doar de amintire. Fiecare intră în vectorul lui. Importanța e o constantă a amintirii, deci intră la fel. τ e un moment de referință fix, numit **ancoră**. Trucul e cunoscut din baze de date ca **forward decay** (Cormode et al., 2009).
- **„Indexul pliat” (folded).** Numele din raport pentru indexul care conține vectorii cu cele două coordonate în plus, adică scorul „pliat” în produsul scalar.
- **Ancoră și segment.** Dacă ancora e departe în trecut, coordonata de recență a amintirilor noi devine uriașă, iar graful se strică. De aceea memoria e împărțită în **segmente**, fiecare cu ancora lui, așa încât coordonatele rămân mici. Segmentul în care intră amintirile noi e cel **cald**. Se închide după o săptămână.
- **Compactare.** Când se strâng prea multe segmente închise, le unești într-unul și recalculezi coordonatele față de o ancoră nouă. Așa fac și bazele de date de tip LSM (RocksDB, Cassandra).
- **Buffer și marcaj de ștergere (tombstone).** Când o amintire e regăsită, recența ei se schimbă, deci vectorul ei nu mai e bun. O marchezi ca ștearsă acolo unde era și o pui într-o listă mică (256 de amintiri) pe care o scanezi exact la fiecare întrebare. Când lista se umple, amintirile intră în segmentul cald.
- **Factorul `s`.** Cât de mult contează cele două coordonate noi când se construiește graful. Rezultatul produsului scalar nu depinde de `s`, doar forma grafului.
- **Rescorare.** La final, candidații găsiți de index sunt recalculați cu formula completă și ordonați.
- **Curbă de uitare.** Cât de mult mai contează o amintire după un anumit timp. În articol e o exponențială (0,995 pe oră). În psihologie, uitarea urmează mai degrabă o putere a timpului: scade repede la început, apoi foarte încet.
- **Funcție complet monotonă, teorema lui Bernstein.** O curbă care scade, cu o pantă care se îndulcește, și așa mai departe la toate derivatele. Bernstein a arătat că orice astfel de curbă e un amestec de exponențiale. De aici vine a doua extensie: puterea din ACT-R se poate scrie aproape exact cu 8 exponențiale.
- **Front Pareto.** Dintr-o mulțime de vectori, cei pe care nu îi întrece niciun alt vector la toate coordonatele deodată. Maximul unui produs scalar cu un vector cu toate componentele pozitive e mereu pe frontul Pareto, deci ajunge să te uiți la el.
- **Recall-ul din proiect.** Ce fracțiune din top-k ales de scorul exact din articol întoarce fiecare metodă. 1 înseamnă că agentul își amintește exact ce și-ar fi amintit cu scanarea completă.
- **Cele patru metode comparate:** scanarea exactă (referința), indexul tău, overfetch cu re-rank (LangChain, K=100) și doar relevanța (căutare vectorială simplă, fără recență și importanță).
- **Willowbrook.** Orașul tău simulat cu 25 de agenți, cu reguli în loc de LLM, ca să poți rula sute de simulări identice și gratuite.
- **Seed.** Numărul de pornire al generatorului aleator. Același seed dă aceeași lume, deci poți rula aceeași simulare cu toate cele patru metode și compara diferențele.
- **Statistica.** **Testul Wilcoxon pe perechi** compară două metode pe aceleași seed-uri, fără să presupună că datele sunt normale. **Corecția Holm** ajustează pragul pentru că ai făcut 36 de comparații (altfel unele ar ieși „semnificative” din întâmplare). **Intervalul de încredere de 95%** e zona în care e probabil să fie media adevărată.

### Ce rezolvă proiectul, pe scurt

Problema: agenții cu memorie mare își aleg amintirile după un scor pe care bazele vectoriale nu îl pot căuta, deci în practică se folosește o aproximare (overfetch) care pierde aproape tot top-ul real când memoria crește. Soluția: scrii scorul ca produs scalar și îl cauți direct cu HNSW, cu inginerie pentru timp (segmente), refresh (buffer) și normalizare. Dovada: recall aproape 1 la un milion de amintiri, de aproximativ o sută de ori mai rapid decât scanarea, iar în oraș comportamentul agenților rămâne cel al regăsirii exacte, pe când metodele aproximative îl schimbă. A doua extensie duce ideea mai departe: și curbele de uitare din psihologie, de exemplu puterea din ACT-R în care contează fiecare folosire a unei amintiri, se aproximează cu 8 exponențiale, deci se caută la fel, cu un singur index.

---

## 1. Povestea în 60 de secunde

Dacă te întreabă cineva „despre ce e proiectul tău?”, răspunsul e ăsta:

> HNSW e indexul pe care îl folosesc aproape toate bazele de date vectoriale ca să găsească rapid cei mai asemănători vectori. Agenții generativi din articolul lui Park (Generative Agents) își aleg însă amintirile după un scor care combină recența, importanța și relevanța, nu doar asemănarea. De aceea framework-urile precum LangChain iau primele 100 de amintiri după asemănare și le reordonează, iar orice amintire din afara celor 100 nu e evaluată deloc, oricât de recentă sau importantă ar fi. Pentru că recența scade exponențial, scorul agentului se poate scrie exact ca un produs scalar între doi vectori cu două coordonate în plus. Deci un singur index HNSW cu produs scalar întoarce direct amintirile cu cel mai mare scor real. Aceeași identitate, pentru un scor mai simplu, a publicat-o independent în septembrie 2026 și un grup de cercetători (TANGO). Eu am construit în jurul ei o memorie completă pentru agenți și am extins-o la curbele de uitare din psihologie: și scorul din ACT-R, unde contează fiecare folosire a unei amintiri, devine un singur produs scalar, cu 8 coordonate în plus. Am testat pe până la un milion de amintiri și într-o simulare cu 25 de agenți, iar pe lângă asta am reimplementat HNSW după pseudocodul din articol și am găsit o inversare a recenței în codul public al Generative Agents.

---

## 2. Articolul: ce trebuie să știi pe de rost

### Problema
Ai `n` vectori (de exemplu embedding-uri de text) și o interogare. Vrei cei mai apropiați `k` vectori. Varianta exactă compară interogarea cu toți vectorii, deci costul crește liniar cu `n`. La un milion de vectori de 100 de dimensiuni înseamnă zeci de milisecunde pe interogare, pe un nucleu. Structurile clasice (arbori kd) nu ajută în dimensiuni mari. Soluția: căutare **aproximativă**, măsurată prin **recall@k** (ce fracțiune din adevărații `k` vecini ai găsit).

### NSW, ideea de la care pornește
Un graf în care fiecare nod e legat de vecinii lui apropiați. Cauți **lacom**: pleci dintr-un nod, te muți mereu la vecinul cel mai apropiat de interogare, te oprești când niciun vecin nu e mai aproape. Dacă inserezi elementele pe rând, primele noduri primesc legături lungi, care devin „scurtături”. Problema NSW: numărul de pași crește tot polilogaritmic, iar nodurile vechi devin hub-uri cu foarte multe legături.

### HNSW, ideea centrală
Separă legăturile pe **niveluri**, exact ca un **skip list**. Fiecare element primește un nivel maxim `l = ⌊−ln(U) · mL⌋`, cu `U` uniform. Asta face ca numărul de elemente să scadă exponențial de la un nivel la altul. Sus sunt puține elemente și legături lungi, jos sunt toate elementele și legături scurte.

- **Căutarea (Alg. 2 și 5):** pleci din punctul de intrare de pe nivelul de sus, cobori lacom (fascicul de lățime 1) până la nivelul 1, apoi pe nivelul 0 faci o căutare best-first cu un fascicul de `ef` candidați.
- **Inserarea (Alg. 1):** aceeași căutare, apoi legi elementul nou de `M` vecini pe fiecare nivel al lui.
- **Euristica de vecini (Alg. 4):** nu iei pur și simplu cei mai apropiați `M`. Accepți un candidat doar dacă e mai aproape de elementul nou decât de oricare vecin deja acceptat. Așa legăturile pleacă în direcții diferite, iar clusterele rămân conectate între ele.
- **Parametri:** `M` (legături per nod; `2M` pe nivelul 0), `efConstruction` (cât de atent construiești), `ef` (cât de atent cauți), `mL = 1/ln M`.
- **Complexitate:** în condiții idealizate, numărul de evaluări de distanță per interogare crește ca `O(log n)`.

### De ce a avut impact
A fost cel mai bun algoritm din benchmark-urile ANN (ANN-Benchmarks) ani la rând și e implementat în aproape toate bazele de date vectoriale (pgvector, Qdrant, Milvus, Weaviate, Elasticsearch). Biblioteca autorilor e `hnswlib`.

### Limite (dacă te întreabă)
- Graful ocupă memorie (pe lângă vectori, `2M` legături per nod).
- Ștergerile sunt incomode: în hnswlib marchezi elementul ca șters, iar nodul rămâne în graf.
- Construcția e mai lentă decât la IVF, iar căutarea e greu de împărțit pe mai multe mașini.
- Garanția `O(log n)` vine din ipoteze idealizate (grafuri Delaunay), nu e demonstrată pentru date reale de dimensiuni mari.

---

## 3. Reimplementarea și ce ai verificat

- Am scris Algoritmii 1-5 de la zero în Python compilat cu numba, cât mai aproape de pseudocod (`agentmem/paper_hnsw.py`).
- **De ce numeri evaluări de distanță?** Pentru că timpul depinde de mașină și de limbaj, iar numărul de distanțe calculate nu. E o măsură de cost corectă pentru comparații între variante.
- **Validare:** pe GloVe-100 (500 000 de vectori), cu aceiași parametri ca hnswlib, recall-ul e practic același la orice `ef`. hnswlib e de câteva ori mai rapid, pentru că e C++ cu instrucțiuni SIMD.
- **Ce am reprodus din studiul de parametri al articolului:**
  - în dimensiuni mici (d=4, un milion de puncte) ierarhia ajută mult: pentru recall 0,9 graful plat are nevoie de 211 distanțe pe interogare, cel ierarhic de 95 cu `mL` implicit și de 79 cu cel mai bun `mL`;
  - în dimensiuni mari (d=1024) aproape că nu contează: orice variantă verifică aproape jumătate din puncte;
  - pe date cu 100 de clustere izolate, selecția simplă de vecini se oprește la recall 0,765, iar euristica ajunge la 0,996;
  - pe GloVe, graful plat e la fel de bun ca cel ierarhic (5 066 față de 5 101 distanțe pentru recall 0,9). Asta e în acord cu un articol din 2024, „Down with the Hierarchy”: în dimensiuni mari, câteva noduri hub formează o „autostradă” care face treaba nivelurilor de sus. Euristica contează și aici: fără ea ai nevoie de 27–34% mai multe distanțe.

**Dacă te întreabă de ce e diferit de hnswlib:** la inserare, articolul trimite toată lista de rezultate `W` ca puncte de intrare pentru nivelul următor, hnswlib doar cel mai apropiat element. Eu am urmat articolul.

---

## 4. Extensia: ce trebuie să poți explica la tablă

### Scorul din Generative Agents
`scor = a·recență + b·importanță + c·relevanță`, fiecare termen scalat min-max la [0, 1], cu `a = b = c = 1` în articol.
- recență = `0,995^(ore de la ultimul acces)`
- importanță = nota 1-10 dată de LLM când se creează amintirea
- relevanță = similaritatea cosinus dintre embedding-uri
- o amintire regăsită primește timpul curent ca „ultimul acces” (refresh)

### Problema
Indexul vectorial știe doar relevanța. LangChain (`TimeWeightedVectorStoreRetriever`) ia primele 100 după relevanță plus ultimele `k` create și le reordonează. Restul nu sunt evaluate deloc. Pe slide-ul cu Max Phillips: 26 din cele 30 de amintiri pe care le-ar alege agentul din articol nu ajung nici măcar candidate.

### Ideea (3 rânduri, învață-le)
1. Recența e exponențială, deci se desparte în jurul oricărui moment de referință τ: `δ^(T−t) = δ^(T−τ) · δ^(τ−t)`. Primul factor depinde doar de interogare, al doilea doar de amintire. (Asta e „forward decay”, Cormode et al., 2009.)
2. Pun în vectorul amintirii `[e, s·δ^(τ−t), s·importanță]`, iar în vectorul interogării `[c·q, (a/s)·δ^(T−τ), b/s]`.
3. Produsul scalar dintre ei dă **exact** scorul. Deci top-k după scor e o căutare după produs scalar maxim (MIPS), pe care HNSW o poate face direct.

**Normalizarea min-max:** scade minimul (o constantă, nu schimbă ordinea) și împarte la amplitudine (doar rescalează ponderile). Deci rămâne tot un produs scalar, cu ponderile `a/R_rec`, `b/R_imp`, `c/R_rel`. Amplitudinile de recență și importanță le ții la zi ieftin. Pe cea de relevanță o estimezi cu încă două căutări: cel mai bun match pentru `q` și pentru `−q`.

### Ce a trebuit rezolvat ca să meargă în practică
- **Coordonate mărginite.** Dacă ancora τ e în trecut, `δ^(τ−t)` crește exponențial (de ~50 000 de ori în 90 de zile) și graful se strică. Cu ancora la cel mai vechi acces, recall-ul cade la 0,06 pe GloVe și la 0,20 pe memoriile orașului, indiferent de `ef`. Soluția: **segmente** cu ancoră proprie, un segment „cald” care se închide după o săptămână, iar segmentele vechi se unesc și se re-ancorează (ca la arborii LSM din baze de date).
- **Refresh la acces.** O amintire regăsită își schimbă coordonata de recență. O marchez ștearsă în segmentul ei și o mut într-un buffer mic (256) care e scanat exact. Când bufferul se umple, intră în segmentul cald.
- **Factorul de construcție `s`.** Produsul scalar e exact pentru orice `s`, dar graful e construit pe vectori, deci `s` decide cât contează metadatele în vecinătăți. Prea mic: graful nu „vede” timpul. Prea mare: graful grupează după vârstă, nu după sens. Dintre valorile încercate, `s = 1` e cel mai bun pentru ponderile din articol. Când relevanța domină, `s = 0,5` e mai sigur.
- **Rescorare exactă** a candidaților la final, cu formula completă. Fără normalizare (ca la LangChain), aproximarea poate doar să piardă amintiri. Cu min-max, candidații sunt ordonați cu ponderile obținute din amplitudinile estimate, deci ordinea poate ieși puțin diferită de cea exactă. De aici vine cea mai mare parte din ce lipsește până la recall 1.

### Cât de originală e ideea
Bucățile sunt cunoscute: forward decay e din 2009, căutarea după produs scalar cu grafuri e din 2018 (Morozov și Babenko). Mai mult, identitatea însăși a fost publicată pe 1 septembrie 2026, cu șase săptămâni înaintea raportului, de Wei și colegii lor (OceanBase, Ant Group și Université Paris Cité), sub numele STR, într-un articol numit TANGO. Ei au doar similaritate plus o exponențială a momentului creării, fără importanță, fără normalizare și fără amintiri al căror timp se schimbă. Spune deschis: „am ajuns la ea independent, dar ei au publicat primii, așa că îi citez”.

Ce rămâne al tău: scorul complet al agenților (importanță, normalizare min-max), memoria vie (reîmprospătare, segmente, ancore), măsurarea efectivă a cât pierde re-ranking-ul (TANGO doar afirmă că pierde), efectul asupra comportamentului agenților, bug-ul din cod și a doua extensie, cu curbele de uitare (secțiunea 6). Pentru aceasta din urmă n-am găsit nimic publicat, dar spune „din câte am găsit”, nu „sigur nimeni”.

---

## 5. Experimentele: ce ai măsurat și ce a ieșit

### Benchmark static
- **Date:** vectori GloVe (aceiași ca în ANN-Benchmarks) cu metadate de agent sintetice, până la 1 000 000 de amintiri; plus amintirile din simulare (cele 25 de agenți după 92 de zile).
- **Rezultat:** cu ponderile din articol, la un milion de amintiri, relevance-only găsește 0,001 din adevăratul top-10, overfetch cu K=100 găsește 0,006 (cu K=1000, 0,040). Cu K=100 rămâne mic la orice mărime (0,045, 0,068 și 0,006 la 10 000, 100 000 și un milion). Indexul pliat găsește 0,975 în 0,53 ms. Scanarea exactă durează 64,5 ms când NumPy calculează toată formula și vreo 46 ms pentru un singur produs matrice-vector, deci indexul e de 87 până la 121 de ori mai rapid (spune „de aproximativ o sută de ori”). Pe memoriile din simulare (126 454): indexul pliat 0,96, overfetch-ul 0,28, relevance-only 0,04.
- **De unde vine ce lipsește până la 1:** mai ales din estimarea amplitudinii relevanței (cele două căutări mici după `q` și `−q`). Cu amplitudinea exactă, același index ajunge la 0,997. Dacă faci căutările de estimare mai adânci (fascicul 128 în loc de 32), urci la 0,988, dar interogarea devine de vreo 2,5 ori mai lentă. E un buton, nu o limită a metodei.
- **Variația între construcții:** hnswlib construiește pe două fire, deci două construcții ale aceluiași index dau grafuri puțin diferite. Pe memoriile orașului, 7 construcții au dat între 0,959 și 0,968. Diferențele de ordinul 0,01 între tabele sunt zgomot.
- **Când NU ajută:** dacă relevanța domină scorul (pondere de câteva ori mai mare decât restul), primele 100 după relevanță conțin deja aproape tot top-ul, iar overfetch-ul ajunge. Atunci indexul pliat are nevoie de `ef` mai mare. Spune asta singur, arată că știi limitele.
- **La memorii mici:** la memoria unui singur agent din oraș (câteva mii de amintiri), scanarea exactă e chiar mai rapidă decât indexul meu scris în Python (1,77 față de 1,90 ms). În benchmark, indexul e mai rapid de la 10 000 de amintiri în sus. Iar când memoria are cel mult K + k amintiri, overfetch-ul vede tot și e exact, deci nici câștigul de corectitudine nu există acolo.

### LangChain
Am comparat cu clasa reală din LangChain pe memoriile a 5 agenți după 30 de zile, cu rata de uitare 0,005 pe oră, ca recența să fie cea din articol (implicit, LangChain are 0,01). Față de top-10 exact al formulei LangChain, retriever-ul lor găsește cam jumătate, al meu tot. Nu spune că e un înlocuitor complet: are aceeași formulă și câmpurile principale, dar primește un model de embedding în loc de o bază de date vectorială.

### Simularea Willowbrook
- 25 de agenți pe reguli, cu bucla din Generative Agents (percepție, conversații, planificare), fiecare apel de LLM înlocuit cu o regulă.
- **Calibrare:** trei parametri nu există în articol. Probabilitatea de a veni la petrecere am fixat-o la 0,45, după cei 5 din 12 invitați din articol. Pe ceilalți doi (cât de des vorbesc, cât de des dau mai departe o veste) i-am ales dintr-o grilă de 15 combinații, astfel încât, cu regăsire exactă și un oraș „tânăr”, rezultatul să semene cu articolul: 13 din 25 au aflat de petrecere, 5 din 12 invitați au venit. Am obținut 13,4 și 5,1 în medie.
- **Experimentul:** aceeași petrecere, dar în orașe care au trăit deja 0, 7, 30 sau 90 de zile când află Maria de ea (rulări de 2, 9, 32 sau 92 de zile, petrecerea e mereu în ultima zi), deci agenții au tot mai multe amintiri. Pentru fiecare variantă, 30 de seed-uri și 4 metode de regăsire.
- **Ce a ieșit** (medii pe 30 de seed-uri; tabelul complet e în raport):
  - **orașul tânăr (rularea de 2 zile, ~100 de amintiri):** toate metodele dau practic același rezultat;
  - **indexul meu:** același rezultat ca scanarea exactă, identic pe toate seed-urile până la 32 de zile și pe 26 din 30 la 92 de zile (pe celelalte 4, diferențe de cel mult 2 agenți, în ambele sensuri, iar mediile rămân practic egale);
  - **doar relevanța:** de la 9 zile încolo, cu aproximativ 40% mai puțini agenți află de petrecere și vin (la 92 de zile: 8,2 în loc de 13,1 află, 3,3 în loc de 5,8 vin). Invitația circulă prin conversații, iar întrebarea către memorie într-o conversație e numele celuilalt, cu care petrecerea nu prea seamănă;
  - **overfetch (LangChain):** la 9 zile răspândește vestea puțin mai slab decât scanarea exactă (11,2 față de 11,7, semnificativ), la 32 de zile diferența e în limita zgomotului, iar la 92 de zile clar mai bine (15,2 în loc de 13,1 află), deși găsește sub jumătate din amintirile pe care le-ar alege scorul din articol.
- **De ce overfetch-ul răspândește mai repede vestea la 92 de zile:** LangChain pune mereu printre candidați și cele mai noi amintiri create. O veste tocmai aflată e printre ele, deci ajunge mereu candidată. Concurenții ei din scorul real (observații recente, dar nu chiar cele mai noi și nu foarte asemănătoare cu întrebarea) nu intră deloc printre candidați. Deci vestea urcă mai ușor decât ar urca în scorul real. **Am verificat:** fără cele mai noi amintiri printre candidați, la 92 de zile rezultatul cade la 8,3 agenți care află și 3,8 care vin, aproape cât la doar-relevanță, deși procentul de amintiri corecte găsite rămâne aproape același (0,50 față de 0,49).
- **Concluzia pe care s-o spui:** o aproximare nu pierde informație la întâmplare, ci schimbă ce informație contează, iar direcția depinde de detalii (ce amintiri noi intră printre candidați, cât de mare e memoria). Iar recall-ul singur nu prezice efectul: două metode cu același recall au dus orașul la rezultate foarte diferite.
- **Statistică:** comparații pe perechi (același seed, aceeași lume), cu testul Wilcoxon și corecția Holm pentru cele 36 de comparații din tabel. Intervalele de încredere folosesc distribuția t.

### Bug-ul din codul public Generative Agents
- În `retrieve.py`, amintirile sunt sortate **crescător** după ultimul acces și primesc recența `0,995^poziție` (0,995 pentru cei 25 de agenți din scenariul principal, 0,99 în configurația implicită). Deci cea accesată cel mai demult primește recența maximă, iar cea mai recentă primește 0 după scalare.
- În plus, recența depinde de poziție, nu de ore, iar ponderile sunt 0,5 / 3 / 2, nu 1 / 1 / 1.
- **Cum am verificat:** am rulat funcția lor `new_retrieve` nemodificată pe personaje simulate, doar apelul la OpenAI pentru embedding l-am înlocuit.
- **Cât contează:** ca să izolezi inversarea, compari cu aceleași ponderi din cod, dar cu recența corectă. Top-30 coincide în proporție de 95%, dar amintirile alese au fost accesate în medie acum 17,6 ore în loc de 10,0 ore, adică de aproape două ori mai demult. (Față de formula completă din articol: 89% și 8,6 ore, dar acolo intră și diferența de ponderi.)
- **Atenție la formulare:** nu spune că rezultatele din articol sunt greșite. Spune că oricine refolosește codul public primește altă recență decât cea descrisă în articol. Nu am găsit pe nimeni care să fi raportat asta public, dar pagina de issues n-am putut-o căuta direct, deci uită-te tu acolo înainte de prezentare.
- **Codul mai nou al autorilor** (genagents, 2024, cel pentru simularea cu 1 000 de oameni) calculează recența corect, din pașii de timp, și îi dă implicit ponderea 0. Dacă te întreabă, spune asta: bug-ul e doar în repository-ul original, care e însă cel pe care îl refolosește toată lumea.

---

## 6. A doua extensie: curbele de uitare

### Problema

Până aici recența a fost cea din articol: o exponențială a ultimului acces, care se înjumătățește la 138 de ore și după o lună e practic zero. Psihologia memoriei descrie altfel uitarea.

- **Anderson și Schooler (1991)** au măsurat cât de des revine o informație în viața reală (titluri din New York Times, ce le spun părinții copiilor mici, e-mailurile primite de unul dintre autori). Au găsit că probabilitatea să ai iar nevoie de ea scade ca o **putere** a timpului trecut de la fiecare apariție și crește cu **numărul** de apariții. Uitarea la oameni urmează tot o putere (Wixted și Ebbesen, 1991).
- **ACT-R** (arhitectura cognitivă a lui John Anderson, de la Carnegie Mellon) transformă asta în activarea de bază a unei amintiri: `B = ln Σ (T − t_j)^(−0,5)`, o sumă peste **toate** folosirile ei.
- Memoriile pentru agenți au început să împrumute astfel de curbe. MemoryBank întărește o amintire de fiecare dată când e reamintită, iar FadeMem o face să se estompeze după o exponențială „întinsă”. Niciunul nu spune cum cauți repede într-o memorie mare cu un astfel de scor. TANGO acceptă doar o exponențială a momentului creării.
- **De ce nu merge direct trucul din secțiunea 4:** exponențiala se desparte în jurul unei ancore (`δ^(T−t) = δ^(T−τ) · δ^(τ−t)`), puterea nu. `(T − t)^(−0,5)` nu se poate scrie ca produs între ceva ce depinde doar de `T` și ceva ce depinde doar de `t`.

### Scorul folosit

`scor = D + importanță + relevanță`, fiecare termen scalat min-max la [0, 1] și cu ponderi 1, ca în articol. Noul termen e

`D = Σ (1 + Δ / 1 h)^(−0,5)`, suma peste toate momentele în care amintirea a fost creată sau regăsită (`Δ` e vârsta fiecărei folosiri).

- `0,5` e valoarea obișnuită din ACT-R. `1 + Δ / 1 h` e acolo doar ca să nu fie infinit la vârsta 0.
- Fără logaritmul din ACT-R, pentru că logaritmul unui produs scalar nu mai e un produs scalar.
- A doua variantă păstrează doar ultima folosire, `D = (1 + Δ / 1 h)^(−0,5)`, ca singura diferență față de articol să fie curba.

### Ideea (3 rânduri, învață-le)

1. Puterea, curba hiperbolică și exponențiala întinsă sunt **complet monotone**, deci, după teorema lui Bernstein (1928), sunt amestecuri de exponențiale. Ajung câteva exponențiale bine alese: **8** urmează puterea din ACT-R pe un an întreg cu o eroare de cel mult **0,30% din amplitudine**. Ponderile le găsesc cu cele mai mici pătrate, cu condiția să fie pozitive (NNLS).
2. Fiecare exponențială se pliază exact ca înainte. Amintirea primește **8 coordonate de timp** în loc de una, iar întrebarea 8 factori `e^(−λ_r·(T − τ))`. Scorul e tot un singur produs scalar, acum cu `d + 8 + 1` coordonate.
3. **Frecvența e gratuită.** Amintirea ține 8 numere `y_r` și momentul ultimei folosiri. La o folosire nouă: `y_r ← y_r · e^(−λ_r·(t − t_ultim)) + 1`. Nu trebuie păstrată toată istoria. Scorul din articol e cazul cu o singură exponențială și `y = 1`.

**Ce garanție ai:** dacă scorul aproximat al fiecărei amintiri e la cel mult `η` de cel adevărat, orice amintire din top-k aproximat are un scor adevărat la cel mult `2η` de al k-lea cel mai bun. În benchmark, scanarea exactă cu curba aproximată întoarce **0,996** din adevăratul top-10, deci aproximarea curbei aproape nu costă.

### Ce a trebuit schimbat la memorie

- **Ratele sunt foarte diferite.** Cea mai rapidă dintre cele 8 exponențiale are 1,7 pe oră, deci coordonata unei amintiri folosite după ancoră crește de 5,5 ori pe oră. Segmentul cald din secțiunea 4 (ancorat când se deschide, umplut o săptămână) ar ajunge la coordonate de ordinul 10^17 după o singură zi.
- **Soluția: partea recentă.** O amintire stă într-o listă scanată exact cât timp ultima ei folosire e mai nouă de **W = 24 de ore**. Abia apoi intră în segmentul cald. Segmentul cald e deschis `L ≤ W` ore și are ancora `τ = deschidere + L − W`, adică puțin în trecut. Așa, orice amintire dintr-un segment a fost folosită ultima oară înainte de ancora lui. Toate coordonatele stau între 0 și `w_r·y_r`, iar factorii întrebării între 0 și 1, oricât de rapidă ar fi rata.
- **Refresh-ul** e mai simplu: o amintire regăsită are ultima folosire „acum”, deci se întoarce în partea recentă. Partea recentă ține locul bufferului de 256 de dinainte.
- **Amplitudinea lui D pentru min-max, fără nicio căutare.** Factorii întrebării sunt toți pozitivi, deci maximul lui `D` e atins pe **frontul Pareto** al vectorilor de timp (amintirile pe care nu le întrece alta la toate cele 8 coordonate deodată), iar minimul pe frontul invers. Fronturile sunt mici: din 94 037 de amintiri din cel mai mare segment, 26 sunt pe frontul de maxim și una pe cel de minim. Deci amplitudinea lui `D` e exactă. Pe cea a relevanței o estimez tot cu căutările după `q` și `−q`.

### Rezultate pe un milion de amintiri

- **Date:** aceiași vectori GloVe, fiecare amintire cu o istorie de folosire sintetică: creată la un moment uniform pe 90 de zile, apoi folosită din nou de un număr geometric de ori (1,2 în medie).
- **Indexul pliat:** 0,982 din adevăratul top-10 la 100 000 de amintiri (0,70 ms) și **0,969 la un milion (2,40 ms)**. Cu o căutare mai adâncă, 0,975 în 2,87 ms.
- **Overfetch cu K = 100: 0,030** la un milion (cu K = 1000: 0,101), deși primește amplitudinile exacte. **Doar relevanța: 0,008.**
- **Timpul:** scanarea exactă din codul meu durează 268 ms la un milion, pentru că evaluează 8 exponențiale pentru fiecare amintire. Comparația corectă e cu cea mai ieftină scanare posibilă, un produs matrice-vector de vreo 46 ms, deci indexul e de vreo 20 de ori mai rapid. Spune „de vreo 20 de ori”, nu „de 112 ori”.
- **Unde se duce timpul:** în scanarea părții recente, care aici are 60 824 de amintiri (cele folosite în ultimele 24 de ore). Cu W = 6 ore rămân 19 059, iar interogarea scade la 1,55 ms, cu recall 0,954. E un buton, ca `ef`.
- **Câte exponențiale:** la 100 000 de amintiri recall-ul e 0,958, 0,977, 0,988 și 0,991 pentru 4, 6, 8 și 10 exponențiale. Peste 8 aproape că nu mai câștigi nimic.
- **Curba chiar schimbă ce își amintește agentul:** la un milion de amintiri, doar 17% din top-10 cu scorul ACT-R e și în top-10 cu scorul din articol (34% la 100 000). Care curbă e mai potrivită pentru agenți e altă întrebare, la care proiectul nu răspunde. Ideea e că indexul nu mai decide în locul tău.

### În oraș

Rulările din oraș cu curba ACT-R sunt încă în desfășurare. Rezultatele vin aici când se termină.

### Întrebări probabile despre curbe

**„E nou?”** Din ce am găsit, da. TANGO are doar exponențiala momentului creării, iar MemoryBank și FadeMem nu discută căutarea. Nu am găsit pe nimeni care să aproximeze curba din ACT-R cu exponențiale ca s-o caute cu un index vectorial. Matematica e veche (Bernstein, 1928, apoi sumele de exponențiale ale lui Beylkin și Monzón, 2005). Nou e felul în care le-am pus împreună și memoria care face asta să meargă. Spune „din câte am găsit”, nu „sigur nimeni”.

**„De ce suma și nu logaritmul, ca în ACT-R?”** Logaritmul nu schimbă ordinea amintirilor după `D`, dar schimbă cât cântărește `D` față de importanță și relevanță. Iar logaritmul unui produs scalar nu e un produs scalar, deci nu se mai poate căuta direct. Suma favorizează ceva mai mult amintirile folosite des. Am scris asta la limite.

**„De ce 8 exponențiale?”** Eroarea scade repede: cu 8 e 0,30% din amplitudine pe un an întreg. Recall-ul crește până la 8 și aproape deloc după (0,988 la 8, 0,991 la 10).

**„Merge pentru orice curbă?”** Pentru orice curbă complet monotonă: puterea, hiperbola, exponențiala întinsă cu exponent cel mult 1. Nu pentru o curbă care urcă din nou sau are praguri. Nici pentru rate de uitare diferite de la o amintire la alta (cum au MemoryBank și FadeMem), decât dacă iau puține valori, cu câte un bloc de coordonate pentru fiecare. Asta n-am implementat.

**„De ce ai nevoie de partea recentă?”** Pentru că cea mai rapidă exponențială crește de 5,5 ori pe oră după ancoră. Fără ea, după o zi coordonatele ajung la 10^17 și graful nu mai găsește nimic. Cu ea, toate coordonatele rămân mici, pentru orice rată.

**„Cât costă în plus?”** 8 coordonate de timp în loc de una per amintire, 8 numere de actualizat la fiecare folosire și scanarea părții recente la fiecare întrebare.

---

## 7. Întrebări probabile și răspunsuri scurte

**„De ce n-ai folosit un LLM real pentru agenți?”**
Ca să pot rula sute de simulări identice (720 în total), gratis și reproductibil. Comparația are nevoie doar ca agenții să acționeze pe baza a ce își amintesc. Am pregătit și o validare cu un LLM local (Ollama) care rulează pe placa mea video. (Dacă ai rulat-o, spune rezultatul.)

**„De ce GloVe și nu un model neural de embedding?”**
Mediul în care am rulat nu avea acces la Hugging Face. Metoda nu depinde de encoder: orice vector unitar funcționează la fel. Codul are deja varianta cu un model neural (MiniLM).

**„Nu e doar un truc matematic?”**
Identitatea e simplă, da. Partea grea e ce urmează: coordonate care pot exploda, amintiri care își schimbă scorul la fiecare acces, normalizarea min-max, alegerea lui `s`. Fără ancorare corectă recall-ul cade la 0,06, iar cu `s` prea mic la 0,26. Plus că am măsurat efectul asupra agenților.

**„De ce nu filtrezi pur și simplu după timp (ultima săptămână)?”**
Pentru că pierzi amintirile vechi dar importante (o nuntă peste o lună, o veste mare), iar pragul e arbitrar. Scorul din articol nu taie nimic, doar ponderează.

**„Cum ar arăta asta într-o bază de date vectorială reală?”**
Adaugi două coordonate la fiecare vector și folosești produsul scalar ca metrică (pgvector, Qdrant, Milvus îl suportă). La refresh faci upsert. Ancorarea și segmentele se pot face cu colecții separate.

**„De ce HNSW și nu FAISS IVF?”**
Am testat și IVF pe aceiași vectori: funcționează, deci metoda nu ține de HNSW. Pe GloVe, HNSW are un compromis viteză-recall mai bun. Pe memoriile orașului, cu ponderile din articol, IVF e chiar puțin mai bun (0,996 la 0,31 ms), probabil pentru că estimează mai bine amplitudinea relevanței. Cu ponderile din cod e însă mult mai slab (0,80 față de 0,98). HNSW e alegerea firească pentru că e deja în toate bazele de date vectoriale, dar recunoaște că nu e singura.

**„De ce nu ai recall 1, dacă formula e exactă?”**
Formula e exactă, căutarea nu: HNSW e un index aproximativ, ca orice index rapid. În plus, normalizarea min-max are nevoie de amplitudinea relevanței, pe care o estimez cu două căutări mici. Cu amplitudinea exactă recall-ul e 0,997, deci aproape tot ce lipsește vine din estimare: o amplitudine estimată greșit schimbă puțin ponderile, deci și ordinea candidaților.

**„Cât costă în plus?”**
Două coordonate per vector și, pentru normalizarea min-max, încă două căutări mici per interogare. La un milion de amintiri, interogarea rămâne sub o milisecundă și ceva.

**„Ce ai face mai departe?”**
Validare cu un LLM real pe mai multe scenarii, embedding-uri neurale, integrare într-o bază de date vectorială reală, parametrii curbei de uitare potriviți pe agenți reali și un studiu despre cum se propagă erorile de regăsire în reflecțiile agenților.

---

## 8. Critica prezentării următoare

Regulamentul spune că după prezentarea ta critici prezentarea următoarei echipe din listă. Întreabă-l pe profesor care e echipa. Dacă e articolul 12 (Generative Agents, Bota Andrei Raul, 19.11), ai deja tot ce-ți trebuie: întreabă cum se calculează concret recența și dacă au văzut diferența dintre articol și cod, și cum ar scala regăsirea la o memorie de luni de zile.

---

## 9. Cum s-a lucrat și cum rulezi tu proiectul

### Ce s-a folosit

- **Totul e Python**, rulat din linia de comandă pe o mașină Linux din cloud, cu 2 nuclee și fără placă video. Nu s-a antrenat nicio rețea neurală. Singurul lucru „învățat” din date sunt vectorii GloVe, pe care cei de la Stanford i-au antrenat demult și pe care codul îi descarcă o singură dată.
- **Bibliotecile**, toate în `requirements.txt`:
  - `numpy` pentru calcule și `numba`, care compilează HNSW-ul scris după articol ca să meargă repede
  - `hnswlib`, HNSW-ul autorilor (C++ cu interfață Python), folosit ca referință și în memoria pliată
  - `faiss-cpu`, doar pentru comparația cu IVF
  - `scipy` și `pandas` pentru statistică și tabele, `matplotlib` pentru figuri
  - `pytest` pentru teste, `langchain-core` și `langchain-classic` pentru comparația cu LangChain
- **Raportul** e scris în LaTeX și compilat cu `latexmk`. Slide-urile sunt făcute în Claude, iar ghidul e un fișier Markdown.

### Cum ajunge un rezultat în raport

1. Fiecare experiment e un script din `experiments/`. Comanda exactă e în README, la „Reproduce”, iar la începutul fiecărui script scrie ce face.
2. Scriptul scrie rezultatele brute în `results/`, de obicei câte un rând de CSV pe rulare.
3. `python -m experiments.plots` desenează figurile din `results/` în `figures/`. `python -m experiments.tables` calculează tabelele și toate cifrele din text, în `report/generated/`. Niciun număr din raport nu e scris de mână.
4. `latexmk` compilează raportul cu ele.

De exemplu, rulările de azi sunt `python -m experiments.exp_town_curve --burns 90 30 --seeds 30`: 360 de simulări ale orașului (2 durate × 2 variante de scor × 3 metode de regăsire × 30 de seed-uri). În fiecare simulare, 25 de agenți trăiesc 32 sau 92 de zile în pași de 15 minute, iar la final scriptul adaugă un rând în `results/town_curve.csv`.

**Nu trebuie să refaci experimentele mari.** Studiul HNSW durează vreo 2 ore, benchmark-ul la un milion de amintiri cam 40 de minute, iar fiecare set de simulări ale orașului cam 3 ore pe 2 nuclee. Rezultatele lor sunt deja în `results/`, iar figurile și cifrele se refac din ele în câteva secunde.

### Cum îl rulezi pe calculatorul tău (Windows)

1. Instalează **Python 3.12** de pe python.org (bifează „Add python.exe to PATH”). Ia 3.12, nu 3.13, pentru că pe Windows unele pachete au variante gata compilate doar pentru 3.12. Instalează și **Git**.
2. Într-un Command Prompt:
   ```
   git clone https://github.com/bogdancolceriu/agentmem-hnsw
   cd agentmem-hnsw
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Dacă `hnswlib` dă o eroare de felul „Microsoft Visual C++ 14.0 or greater is required”, pip încearcă să-l compileze. Ai două variante. Fie instalezi „Build Tools for Visual Studio”, cu componenta „Desktop development with C++”, și reiei comanda. Fie instalezi varianta gata compilată, care are aceeași interfață (n-am testat-o pe Windows):
   ```
   pip install chroma-hnswlib numpy scipy pandas matplotlib faiss-cpu numba pytest langchain-core langchain-classic
   ```
4. Verifică instalarea: `python -m pytest -q tests/` trebuie să arate 42 de teste trecute, în vreo jumătate de minut.
5. **Turul ghidat:** `python demo/tour.py` rulează în 1–2 minute pașii proiectului pe date mici: HNSW-ul din articol față de hnswlib, scorul agenților, plierea în produs scalar, cele patru metode de regăsire și curbele de uitare. La fiecare pas spune ce arată și în ce fișier e codul. `python demo/tour.py --oras` adaugă două simulări scurte ale orașului (prima dată descarcă vectorii GloVe, cam 130 MB).
6. `python -m experiments.plots` și `python -m experiments.tables` refac în câteva secunde figurile și cifrele din raport, din rezultatele salvate.
7. `demo/willowbrook.html`, deschis în browser, reia pas cu pas aceeași petrecere cu trei metode de regăsire.

Cifrele din tur nu sunt cele din raport, pentru că datele lui sunt sintetice și mici. Diferențele dintre metode sunt însă aceleași.

### Cu Claude Code pe calculatorul tău

Ideea e bună. Deschizi Claude Code în folderul proiectului (din aplicația desktop sau din terminal) și îi ceri, de exemplu: „instalează dependențele, rulează testele și `demo/tour.py`, apoi explică-mi fiecare pas”. Se descurcă singur cu erorile de instalare. Apoi îl poți întreba direct despre cod: „unde se construiește vectorul pliat?”, „ce face `_age_out` din `curve_stores.py`?”, „de ce are nevoie memoria de segmente?”. Ce nu poate reface e conversația în care s-a construit proiectul. Pașii mari se văd în commit-urile de pe GitHub, iar fiecare rezultat se reproduce cu comanda lui din README.

Dacă te întreabă cineva cum ai lucrat, spune direct că ai lucrat cu Claude, cu acordul profesorului. La prezentare contează să poți rula și explica fiecare parte, iar pentru asta sunt turul și harta de mai jos.

### Harta codului

| Ce | Unde |
|---|---|
| HNSW după pseudocodul din articol | `agentmem/paper_hnsw.py` (`PaperHNSW`, `search_layer`, `select_heuristic`, `insert`) |
| Scorul agenților și plierea lui | `agentmem/score.py` (`exact_scores`, `fold_memories`, `fold_query`) |
| Memoria pliată, cu segmente și buffer | `agentmem/stores.py` (`FoldedHNSW`) |
| Metodele cu care compari | `agentmem/stores.py` (`ExactStore`, `OverfetchRerank`, `RelevanceOnlyHNSW`) |
| Curbele de uitare ca sumă de exponențiale | `agentmem/curves.py` (`fit_expsum`) |
| Memoria pentru orice curbă | `agentmem/curve_stores.py` (`CurveFolded`) |
| Retriever-ul pentru LangChain | `agentmem/langchain_retriever.py` |
| Embedding-urile (GloVe și SIF) | `agentmem/embed.py` |
| Orașul | `town/world.py` (oamenii, locurile, rutinele), `town/sim.py` (bucla agenților) |
| Experimentele, figurile și tabelele | `experiments/` (un script pe experiment, plus `plots.py` și `tables.py`) |
| Testele | `tests/` |
