Ecco la riscrittura completa, formale e discorsiva del tuo file `README.md`. Questa versione non si limita a elencare le specifiche del motore di Retrieval, ma documenta l'intera evoluzione ingegneristica del codice, spiegando in profondità il **perché** di ogni singola ottimizzazione e correzione introdotta durante lo sviluppo per superare i blocchi infrastrutturali e i limiti degli algoritmi convenzionali.

---

# 📑 Manuale d'Architettura: Motore di Retrieval Ibrido (MedFactCheck)

Il modulo di Retrieval costituisce il nucleo di Information Retrieval (IR) ad alte prestazioni del framework **MedFactCheck**. Il suo obiettivo primario è l'estrazione di evidenze cliniche dense, accurate e prive di rumore a partire da due sorgenti distinte: il corpus statico locale **SciFact** (Ramo KB) e la letteratura medica mondiale aggiornata live tramite l'API di **Europe PMC** (Ramo LIT).

L'intero modulo è orchestrato secondo un design pattern a imbuto (*funnel*) e centralizzato sotto un unico punto d'accesso coerente, ottimizzato per operare in ambienti cloud con risorse hardware vincolate.

---

## 🏗️ 1. Architettura di Sistema e Master Orchestrator

Il sistema adotta il pattern architetturale **Facade** attraverso la classe centralizzata `MedFactCheckRetriever`. Questa scelta risponde alla necessità ingegneristica di disaccoppiare la logica di decomposizione dei claim (gestita dall'LLM a monte) dalle minuzie implementative dei singoli nodi di ricerca.

```
                     [Claim Utente]
                           │
                           ▼
               [MedFactCheckRetriever] (Orchestratore)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    [Ramo KB (SciFact)]         [Ramo LIT (Europe PMC)]
    ├─ BM-25 (Lessicale)        ├─ API Metadata Fetch (Top-50)
    └─ FAISS PQ (Semantico)     ├─ Cache-Aside / In-Memory DB
                                ├─ Chunking Dinamico (20% Overlap)
                                ├─ Filtro BM-25 (Top-100)
                                └─ Reranking BioBERT (Top-20)

```

### Unificazione dei Formati di Output

Nelle prime fasi di sviluppo, il Ramo KB restituiva liste di stringhe grezze, mentre il Ramo LIT produceva dizionari strutturati dotati di metadati e punteggi di confidenza. Questa eterogeneità avrebbe causato il fallimento dei parser semantici dei modelli di ragionamento (Qwen) e classificazione (DeBERTa) a valle.

Il codice è stato standardizzato affinché ogni singolo nodo di retrieval restituisca un formato JSON omogeneo composto da tre chiavi tassative:

* `text`: Lo snippet testuale normalizzato e validato.
* `source`: La tracciabilità esatta della fonte (es. `KB (SciFact - FAISS)` o `PMC ID: XXX`).
* `score`: Il punteggio matematico normalizzato assegnato dall'algoritmo di ranking.

---

## 🔎 2. Il Ramo KB (Knowledge Base Locale)

Il Ramo KB interroga il dataset **SciFact** combinando metodologie di ricerca sparse (lessicali) e dense (semantiche). Per garantire la massima efficienza su Google Colab, la gestione delle risorse è stata riprogettata da zero.

### 2.1 Zero-Latency Embedding e Allineamento NumPy 2.x

Il caricamento iterativo in memoria dei pesi dei modelli linguistici rappresenta uno dei principali colli di bottiglia di I/O nei sistemi RAG di vecchia concezione. MedFactCheck risolve questo problema istanziando il modello asimmetrico `pritamdeka/S-PubMedBert-MS-MARCO` **una sola volta all'avvio dell'applicazione** attraverso la *Dependency Injection*.
Il modello viene forzato in mezza precisione (`torch.float16`), riducendo l'impronta in VRAM da ~440 MB a soli ~220 MB.

**Perché è stato necessario correggere l'ambiente software:** Durante l'aggiornamento dei runtime di Google Colab a NumPy 2.x, le vecchie distribuzioni binarie di FAISS causavano crash sistematici a causa di incompatibilità dell'interfaccia C-API. Il sistema è stato stabilizzato forzando l'installazione di `faiss-gpu-cu12` unificata sotto CUDA 12, garantendo la perfetta convivenza tra NumPy 2.x e l'accelerazione hardware delle matrici di embedding.

### 2.2 Ingestion Fail-Safe e Bypass di Hugging Face

**Perché è stata modificata la data ingestion:** Nelle versioni precedenti, il corpus SciFact veniva importato tramite la funzione `load_dataset` di Hugging Face. Recentemente, la piattaforma Hugging Face ha rimosso il supporto per l'esecuzione di script di caricamento custom interni (`scifact.py`) per ragioni di sicurezza, rendendo obsoleto anche il parametro `trust_remote_code=True`.
Per evitare il blocco totale del software, è stata implementata una pipeline *bulletproof* basata sulla libreria `requests`. Il sistema scarica direttamente il file grezzo `corpus.jsonl` dai mirror ufficiali del benchmark MTEB, effettuando il parsing delle righe JSON in modo nativo e isolando le chiavi `title` e `text` senza dipendere da SDK esterni fragili.

### 2.3 Risoluzione del Type Mismatch FP16/FP32 in FAISS

**Perché è stata inserita la conversione esplicita:** Sebbene l'allineamento dell'embedder globale in `Float16` sia ideale per minimizzare l'uso della VRAM, la libreria open-source FAISS (scritta in C++ nativo) richiede tassativamente array in precisione singola standard a 32-bit (`float32`) per le sue funzioni interne di normalizzazione vettoriale e per l'algoritmo di compressione *Product Quantization* (IndexPQ). Il passaggio diretto dei vettori FP16 sollevava un errore bloccante di tipo `TypeError: argument 3 of type 'float *'`.
La criticità è stata superata iniettando un casting esplicito all'interno delle funzioni di indicizzazione e di ricerca vettoriale (`search_faiss_pq`):

```python
embeddings = embeddings.astype('float32')
faiss.normalize_L2(embeddings)

```

Questa soluzione coniuga i benefici di entrambi i mondi: il modello linguistico risiede stabilmente in VRAM a basso impatto (FP16), mentre il motore geometrico lavora in totale sicurezza matematica (FP32).

### 2.4 Routing Condizionale e OOM-Proofing

A seconda dei parametri logici della query, il sistema devia il flusso informativo:

* **Rotta Lessicale Esatta (`BM-25Okapi`):** Dedicata alla verifica di nomenclature rigide o claim puramente assiomatici, operando mediante un indice invertito residente in RAM.
* **Rotta Semantica Densa (`FAISS PQ`):** Dedicata alla comprensione di sinonimi ed espressioni discorsive, calcolando il prodotto interno (`METRIC_INNER_PRODUCT`) su vettori normalizzati L2 (equivalente algebrico della *Cosine Similarity*).

Per prevenire i crash da *Out Of Memory* (OOM) sui classificatori finali a complessità quadratica $O(n^2)$ (come DeBERTa), è stato inserito un algoritmo di **Hard Truncation Dinamico**. Ogni chunk estratto dalla KB viene troncato rigidamente a un massimo di 500 caratteri utilizzando una funzione `rsplit` in corrispondenza dell'ultimo spazio vuoto, preservando la coerenza sintattica dello snippet ed eliminando il testo ridondante.

---

## 🌍 3. Il Ramo LIT (Letteratura Medica Mondiale)

Il Ramo LIT interroga l'immensa base documentale di **Europe PMC** per validare trial clinici, interazioni farmacologiche o linee guida aggiornate in tempo reale. Dovendo elaborare articoli integrali (*Full-Text*) che superano frequentemente le 10.000 parole, il modulo adotta una rigorosa strategia a imbuto denominata **Massive Recall over Speed**.

```
[50 Paper Integrali da API/DB] ➔ Esplosione in Chunk (150 parole, 20% Overlap) 
                                      │
                                      ▼
                        [Pool di Migliaia di Chunk]
                                      │  (Stadio 1: BM-25 CPU)
                                      ▼
                             [Top-100 Chunk Lessicali]
                                      │  (Stadio 2: BioBERT GPU)
                                      ▼
                             [Top-20 Chunk Semantici]

```

### 3.1 Design Pattern Cache-Aside (Mock DB Adapter)

Per azzerare l'elevata latenza di rete associata al download sequenziale dei testi medici, l'architettura prevede l'integrazione di un database documentale secondo il pattern *Cache-Aside*.
Il sistema esegue una chiamata iniziale leggerissima (`resultType: lite`) per scaricare esclusivamente i metadati e gli ID dei primi 50 articoli pertinenti. Successivamente, esegue un controllo simultaneo tramite un operatore di Bulk-Read (`$in`) per verificare quali documenti siano già presenti localmente.

Nel codice attuale, tale logica è gestita da un **In-Memory Mock DB** (`self.mock_db = {}`), un adapter strutturato in RAM che replica fedelmente l'interfaccia NoSQL e garantisce tempi di risposta a latenza zero per i paper già consultati, pronto per essere mappato su una istanza reale di MongoDB in produzione.

### 3.2 Chunking Granulare ed Esplosione del Pool

I documenti integrali (estratti da cache o scaricati in caso di Cache Miss) vengono atomizzati in blocchi di **150 parole**. Al fine di preservare il nesso causale e clinico delle frasi a cavallo tra due blocchi, viene applicato un **Overlap del 20%** (~30 parole). Questo processo converte i 50 articoli in un pool temporaneo di svariate migliaia di chunk candidati.

### 3.3 Ottimizzazione della Sintassi di Rete e Throttling Anti-Ban

Durante la fase di test, l'interrogazione dell'API esterna di Europe PMC presentava due criticità bloccanti che causavano il fallimento sistematico del download (restituendo 0 chunk totali). Tali problematiche sono state risolte mediante modifiche mirate alla logica di rete:

1. **Rilassamento della Sintassi della Query:** La stringa originale vincolava la ricerca all'esatta corrispondenza testuale racchiudendo il claim tra doppie virgolette (`"{query}"`), riducendo a zero la sensibilità del motore di ricerca letterario. Le virgolette rigide sono state rimosse in favore del raggruppamento logico tramite parentesi tonde `({query}) AND OPEN_ACCESS:y`, massimizzando il recupero (*recall*) semantico dei termini clinici.
2. **Risoluzione del Parsing degli Identificativi:** L'API falliva nel recuperare i testi completi quando l'ID passato era un formato *PMCID* (es. con prefisso PMC) a causa della restrizione forzata del parametro `ext_id:`. La stringa di interrogazione è stata pulita delegando a Europe PMC l'identificazione automatica della natura dell'ID (PMID o PMCID).
3. **Iniezione del Buffer di Throttling:** L'esecuzione ravvicinata di 50 richieste HTTP consecutive esponeva il sistema al blocco temporaneo o permanente dell'indirizzo IP da parte dei firewall dell'EBI per l'attivazione delle difese anti-DDoS (HTTP Error 429 - Too Many Requests). È stato introdotto un cooldown forzato di 100 millisecondi (`time.sleep(0.1)`) subito dopo la gestione di ogni singolo Cache Miss, garantendo la stabilità e la continuità della pipeline nel pieno rispetto delle policy dei server remoti.

### 3.4 Filtrazione a Due Stadi (Funnel Pipeline)

Il pool massivo di migliaia di chunk viene scremato attraverso un processo bifasico sequenziale:

* **Stadio 1 (Filtro CPU Grossolano - Sparse Retrieval):** L'intero database temporaneo viene indicizzato al volo tramite `BM25Okapi`. L'algoritmo calcola la densità lessicale esatta rispetto al claim e seleziona istantaneamente i **top-100 chunk**, scartando tutto il rumore editoriale, metodologico o bibliografico.
* **Stadio 2 (Reranking GPU Fine - Dense Retrieval):** I 100 superstiti vengono inviati in modalità *Batched Inference* all'embedder persistente in mezza precisione. Sfruttando le funzioni vettoriali native di PyTorch (`F.cosine_similarity`) sotto regime di `torch.no_grad()`, viene calcolata la vicinanza concettuale profonda, isolando i **top-20 chunk definitivi**.

Grazie a questa attenta ingegnerizzazione a imbuto, l'LLM a valle (Qwen) riceve un contesto sintetico, ad altissima densità informativa ed epidemiologica, totalmente protetto da fenomeni di allucinazione lessicale e strutturato in modo ideale per avviare le sue routine di *Inferenza Gerarchica Adattiva*.