# 🗄️ Manuale d'Architettura: Storage e Data Provenance (MedFactCheck)

Il modulo di **Storage** è l'infrastruttura di persistenza dati del sistema MedFactCheck. Garantisce che l'intero ciclo di vita di un claim, dall'input iniziale fino al verdetto finale e alle evidenze recuperate, sia memorizzato in modo sicuro, interrogabile e trasparente.

---

## 🏛️ 1. Architettura del Database: MongoDB

Il sistema utilizza **MongoDB** come database NoSQL primario. La scelta di un database orientato ai documenti è perfetta per memorizzare strutture dati complesse e gerarchiche come quelle prodotte dal grafo multi-agente LangGraph.

All'interno del database `medfactcheck`, vengono gestite tre collezioni principali:

### 1.1 `final_results` (Storico dei Claim e Verdetti)
Memorizza il risultato consolidato di ogni verifica. 
- **Struttura:** Contiene il testo originale, l'eventuale immagine, il verdetto finale aggregato (es. *Supported*), il punteggio di confidenza medio, l'array dei *sub-verdicts* (con i relativi ragionamenti CoT) e la traccia completa dell'esecuzione degli agenti (`agent_trace`).
- **Scopo:** Alimenta la cronologia della Dashboard e permette ricerche avanzate tramite Regex e filtri su data e verdetto.

### 1.2 `evidence` (Data Provenance e Tracciabilità)
Salva i singoli passaggi (chunk) recuperati dalla letteratura o dalle Knowledge Base.
- **Struttura:** Ogni documento include il `claim_id` associato, la fonte esatta (es. URL, PubMed ID o DisGeNET), il testo del chunk e il punteggio di similarità calcolato in fase di retrieval.
- **Scopo:** Garantisce la *Data Provenance*. L'utente può sempre risalire alla fonte esatta che ha generato un determinato verdetto, requisito fondamentale in ambito biomedico per combattere le allucinazioni.

### 1.3 `papers` (Cache-Aside della Letteratura)
Funge da layer di caching intelligente per i documenti XML integrali scaricati da Europe PMC.
- **Struttura:** Salva l'ID del paper e l'intero full-text.
- **Scopo:** Evita di scaricare ripetutamente lo stesso paper per claim diversi, azzerando la latenza di rete e proteggendo il sistema dal rate-limiting e dai ban delle API esterne.

---

## 🧠 2. Persistenza dello Stato del Grafo

Per gestire lo stato delle esecuzioni asincrone, il sistema si avvale di `langgraph-checkpoint-mongodb`. Questo meccanismo di "memory saver" cattura lo stato esatto del grafo in ogni singolo step (es. dopo che il Decomposer ha finito, prima che inizi il Retriever).
Questa funzionalità garantisce **Fault-Tolerance**: se il server si riavvia improvvisamente, l'esecuzione del claim riprende senza perdere dati.

---

## 🗃️ 3. Storage Vettoriale (FAISS)
Parallelamente a MongoDB, il sistema mantiene indici vettoriali locali tramite **FAISS** (Facebook AI Similarity Search) per la Knowledge Base strutturata. Gli indici sono salvati su disco con un sistema di **triplo backup automatico** per prevenire corruzioni della cache.