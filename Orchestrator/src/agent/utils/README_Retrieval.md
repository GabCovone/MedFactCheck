# 📑 Manuale d'Architettura: Motore di Retrieval Ibrido (MedFactCheck)

Il modulo di Retrieval costituisce il nucleo di Information Retrieval (IR) ad alte prestazioni del sistema Multi-Agente **MedFactCheck**. Il suo obiettivo primario è l'estrazione di evidenze cliniche dense, accurate e prive di rumore a partire da due sorgenti distinte: la Knowledge Base locale strutturata **DisGeNET** (Ramo KB) e la letteratura medica mondiale aggiornata live tramite l'API di **Europe PMC** (Ramo LIT).

L'intero modulo è orchestrato secondo un design pattern a imbuto (*funnel*) e centralizzato sotto un unico punto d'accesso ottimizzato per operare in ambienti con risorse hardware vincolate.

---

## 🏗️ 1. Architettura di Sistema e Master Orchestrator

Il sistema adotta il pattern architetturale **Facade** attraverso la classe centralizzata `MedFactCheckRetriever`. Questa scelta risponde alla necessità di disaccoppiare la logica di decomposizione dei claim (gestita dall'LLM a monte) dalle minuzie implementative dei singoli nodi di ricerca.

```text
                     [Claim Utente]
                           │
                           ▼
               [MedFactCheckRetriever]
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    [Ramo KB (DisGeNET)]        [Ramo LIT (Europe PMC)]
    ├─ BM-25 (Lessicale)        ├─ API Metadata Fetch (Top-50)
    └─ FAISS PQ (Semantico)     ├─ Cache-Aside / MongoDB
                                ├─ Chunking Dinamico (20% Overlap)
                                ├─ Filtro BM-25 (Top-100)
                                └─ Reranking BioBERT (Top-20)
```

### Unificazione dei Formati di Output

Nelle prime fasi di sviluppo, il Ramo KB restituiva liste di stringhe grezze, mentre il Ramo LIT produceva dizionari strutturati dotati di metadati e punteggi di confidenza. Questa eterogeneità avrebbe causato il fallimento dei parser semantici dei modelli di ragionamento (Qwen) e classificazione (DeBERTa) a valle.

Il codice è stato standardizzato affinché ogni singolo nodo di retrieval restituisca un formato JSON omogeneo composto da tre chiavi tassative:

* `text`: Lo snippet testuale normalizzato e validato.
* `source`: La tracciabilità esatta della fonte (es. `KB (DisGeNET - FAISS)` o `PMC ID: XXX`).
* `score`: Il punteggio matematico normalizzato assegnato dall'algoritmo di ranking.

---

## 🔎 2. Il Ramo KB (Knowledge Base Locale)

Il Ramo KB funge da "oracolo di verità inconfutabile", interrogando la Knowledge Base **DisGeNET** combinando metodologie di ricerca sparse (lessicali) e dense (semantiche). Si occupa di risolvere affermazioni dogmatiche o nozioni statiche.

### 2.1 Zero-Latency Embedding e Allineamento NumPy 2.x

Il caricamento iterativo in memoria dei pesi dei modelli linguistici rappresenta uno dei principali colli di bottiglia di I/O nei sistemi RAG di vecchia concezione. MedFactCheck risolve questo problema istanziando il modello asimmetrico `pritamdeka/S-PubMedBert-MS-MARCO` **una sola volta all'avvio dell'applicazione** attraverso la *Dependency Injection*.
Il modello viene forzato in mezza precisione (`torch.float16`), riducendo l'impronta in VRAM da ~440 MB a soli ~220 MB.

**Perché è stato necessario correggere l'ambiente software:** Durante l'aggiornamento dei runtime di Google Colab a NumPy 2.x, le vecchie distribuzioni binarie di FAISS causavano crash sistematici a causa di incompatibilità dell'interfaccia C-API. Il sistema è stato stabilizzato forzando l'installazione di `faiss-gpu-cu12` unificata sotto CUDA 12, garantendo la perfetta convivenza tra NumPy 2.x e l'accelerazione hardware delle matrici di embedding.

### 2.2 Fonte dei Dati: DisGeNET (Esteso) e Textualizzazione

Il sistema si auto-costruisce interrogando e scaricando i Gold Standard *Curated* da **DisGeNET**:
- **Gene-Disease Associations**: Relazioni validate tra geni e malattie.
- **Variant-Disease Associations (VDAs)**: Relazioni validate tra varianti genetiche (es. SNPs) e patologie.

I dati tabellari TSV originali vengono scaricati (compressi in GZIP), decodificati in tempo reale e convertiti in **frasi strutturate di senso compiuto in inglese** (es. *"The gene BRCA1 is associated with the disease breast cancer..."*). Questo permette ai modelli semantici (Faiss e BM25) di "leggere" una Knowledge Base tabellare come se fosse un libro discorsivo.

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

### 2.5 Gestione Sicura della Cache e Controllo Integrità

Per garantire la massima affidabilità ed evitare crash dovuti a file vettoriali corrotti, il modulo KB implementa un sistema di **Triplo Backup e Controllo d'Integrità**:
- **Validazione Dimensionale**: Al caricamento, il sistema verifica che la dimensione degli indici BM25 (`corpus_size`) e FAISS (`ntotal`) coincida esattamente con il numero di documenti (geni e VDAs) effettivamente scaricati da DisGeNET.
- **Fallback Automatico**: Se viene rilevato un indice obsoleto o corrotto (es. a causa di un'interruzione di corrente durante il salvataggio), il sistema lo scarta e passa automaticamente a uno dei file di backup. Se tutti i backup falliscono, la KB viene re-indicizzata da zero in tempo reale per garantire zero downtime.

---

## 🌍 3. Il Ramo LIT (Letteratura Medica Mondiale)

Il Ramo LIT interroga la base documentale di **Europe PMC** per validare trial clinici, interazioni farmacologiche o linee guida aggiornate in tempo reale. Dovendo elaborare articoli integrali (*Full-Text*) che superano frequentemente le 10.000 parole, il modulo adotta una rigorosa strategia a imbuto denominata **Massive Recall over Speed**.

```text
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

### 3.1 Design Pattern Cache-Aside (MongoDB)

Per azzerare l'elevata latenza di rete associata al download sequenziale dei testi medici, l'architettura si appoggia a **MongoDB** secondo il pattern *Cache-Aside*.
Il sistema esegue una chiamata iniziale leggerissima (`resultType: lite`) per scaricare esclusivamente i metadati e gli ID dei primi 50 articoli pertinenti.

L'istanza MongoDB locale (`medfactcheck.papers`) salva integralmente i file XML estratti. Questo garantisce tempi di risposta a latenza zero per i paper già consultati ed evita il ban delle API. Se un documento non è nel DB, viene scaricato tramite **Thread Pool concorrente** (multi-threading) e salvato asincronamente.

### 3.2 Chunking Granulare ed Esplosione del Pool

I documenti integrali (estratti da cache o scaricati in caso di Cache Miss) vengono atomizzati in blocchi di **150 parole**. Al fine di preservare il nesso causale e clinico delle frasi a cavallo tra due blocchi, viene applicato un **Overlap del 20%** (~30 parole). Questo processo converte i 50 articoli in un pool temporaneo di svariate migliaia di chunk candidati.

### 3.3 Filtrazione a Due Stadi (Funnel Pipeline)

Il pool massivo di migliaia di chunk viene scremato attraverso un processo bifasico sequenziale:

* **Stadio 1 (Filtro CPU Grossolano - Sparse Retrieval):** L'intero database temporaneo viene indicizzato al volo tramite `BM25Okapi`. L'algoritmo calcola la densità lessicale esatta rispetto al claim e seleziona istantaneamente i **top-100 chunk**, scartando tutto il rumore editoriale, metodologico o bibliografico.
* **Stadio 2 (Reranking GPU Fine - Dense Retrieval):** I 100 superstiti vengono inviati in modalità *Batched Inference* all'embedder persistente in mezza precisione. Sfruttando le funzioni vettoriali native di PyTorch (`F.cosine_similarity`) sotto regime di `torch.no_grad()`, viene calcolata la vicinanza concettuale profonda, isolando i **top-20 chunk definitivi**.

Grazie a questa attenta ingegnerizzazione a imbuto, l'LLM a valle (Qwen) riceve un contesto sintetico, ad altissima densità informativa ed epidemiologica, totalmente protetto da fenomeni di allucinazione lessicale e strutturato in modo ideale per avviare le sue routine di *Inferenza Gerarchica Adattiva*.

---

## 🤖 4. Il Retriever Agent (`RetrievalFunc.py`)

L'orchestrazione pratica del recupero è affidata al `RetrieverAgent`. 
Invece di un flusso rigido in cui ogni ricerca esegue tutto, questo agente implementa un pattern **Guided Execution Multi-Tool**:

- Sfruttando le *routes* decise dal `Decomposer` (es. `["kb", "lit"]` o `["kb"]`), mappa i task sui vari Tool specifici (`kb_bm25`, `kb_faiss`, `lit_europe_pmc`).
- Lancia **ricerche parallele** per ogni singolo sub-claim (tramite `ThreadPoolExecutor`), accelerando enormemente il throughput complessivo.
- Dispone di un sistema di **Fallback Globale**: se un sub-claim non trova nulla in una determinata rotta precalcolata dal Decomposer, il sistema espande automaticamente la ricerca in extremis su tutti i database per evitare *false negative*.