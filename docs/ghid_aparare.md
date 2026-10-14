# Ghid de pregătire: prezentare și întrebări

Ghidul ăsta te pregătește pentru prezentarea din 17.12 și pentru întrebările de după. Citește-l o dată de la cap la coadă, apoi revino la secțiunile unde nu ai putea răspunde fără să te uiți. Cifrele exacte sunt în raport (`report/main.pdf`), aici sunt doar cele de care chiar ai nevoie.

---

## 1. Povestea în 60 de secunde

Dacă te întreabă cineva „despre ce e proiectul tău?”, răspunsul e ăsta:

> HNSW e indexul pe care îl folosesc aproape toate bazele de date vectoriale ca să găsească rapid cei mai asemănători vectori. Agenții generativi din articolul lui Park (Generative Agents) își aleg însă amintirile după un scor care combină recența, importanța și relevanța, nu doar asemănarea. De aceea framework-urile precum LangChain iau primele 100 de amintiri după asemănare și le reordonează, iar orice amintire din afara celor 100 nu e evaluată deloc, oricât de recentă sau importantă ar fi. Eu am arătat că, pentru că recența scade exponențial, scorul agentului se poate scrie exact ca un produs scalar între doi vectori cu două coordonate în plus. Deci un singur index HNSW cu produs scalar întoarce direct amintirile cu cel mai mare scor real. Am testat pe până la un milion de amintiri și într-o simulare cu 25 de agenți, iar pe lângă asta am reimplementat HNSW după pseudocodul din articol și am găsit o inversare a recenței în codul public al Generative Agents.

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

### Cât de originală e ideea (răspunde cinstit)
Bucățile sunt cunoscute: forward decay e din 2009, căutarea după produs scalar cu grafuri e din 2018 (Morozov și Babenko). Contribuția mea e combinația aplicată la memoria agenților, partea de inginerie pentru amintiri al căror scor se schimbă la fiecare regăsire și dovada experimentală că alegerea metodei schimbă comportamentul agenților. Nu am găsit lucrarea asta făcută înainte, dar spune „din câte am găsit”, nu „sigur nimeni”.

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
- **Atenție la formulare:** nu spune că rezultatele din articol sunt greșite. Spune că oricine refolosește codul public primește altă recență decât cea descrisă în articol. Înainte de prezentare, caută pe GitHub, la issues, dacă a mai raportat cineva asta (eu n-am avut acces la pagina de issues).

---

## 6. Întrebări probabile și răspunsuri scurte

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
Validare cu un LLM real pe mai multe scenarii, embedding-uri neurale, integrare într-o bază de date vectorială reală și un studiu despre cum se propagă erorile de regăsire în reflecțiile agenților.

---

## 7. Critica prezentării următoare

Regulamentul spune că după prezentarea ta critici prezentarea următoarei echipe din listă. Întreabă-l pe profesor care e echipa. Dacă e articolul 12 (Generative Agents, Bota Andrei Raul, 19.11), ai deja tot ce-ți trebuie: întreabă cum se calculează concret recența și dacă au văzut diferența dintre articol și cod, și cum ar scala regăsirea la o memorie de luni de zile.
