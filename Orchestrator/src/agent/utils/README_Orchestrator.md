# 📑 Manuale d'Architettura: Orchestratore e Supervisore Multi-Agente

Il modulo **Orchestrator** rappresenta il sistema nervoso centrale di **MedFactCheck**. Il suo compito è governare l'intero ciclo di vita della verifica di un claim biomedico (notizia, affermazione o immagine medica), coordinando l'esecuzione asincrona dei vari modelli di Intelligenza Artificiale (LLM, modelli di embedding, cross-encoder), gestendo l'I/O verso il database (MongoDB) e interfacciandosi con il client esterno.

Per garantire la massima flessibilità e scalabilità, l'orchestrazione non si basa su un automa a stati finiti rigido, bensì su un'architettura **Multi-Agente dinamica (Hub & Spoke)** basata sul framework **LangGraph**, esposta nativamente tramite un'interfaccia RESTful (**FastAPI**).

---

## 🏗️ 1. Architettura di Rete e Disaccoppiamento Client-Server

Per simulare in modo accurato un ambiente di produzione *Enterprise* (e permettere il testing ottimale su infrastrutture cloud), il sistema è stato ingegnerizzato separando nettamente il livello di presentazione dal motore di calcolo:

1. **Server Back-End (`api.py` / Orchestrator)**: Un server asincrono ad alte prestazioni basato su **FastAPI** e `uvicorn`. Ha il monopolio esclusivo sull'hardware (allocazione VRAM di Qwen e DeBERTa, caricamento degli indici FAISS, modelli di inferenza). Rimane in ascolto sul path `/verify` accettando richieste HTTP POST asincrone contenenti payload testuali e buffer di immagini multimediali (`UploadFile`).
2. **Client Front-End (`Dashboard.py`)**: Un'interfaccia ultra-leggera in Streamlit che delega il carico computazionale all'API tramite chiamate `requests.post`, interrogando poi localmente MongoDB per renderizzare le metriche e i risultati in tempo reale.

Questo disaccoppiamento risponde direttamente ai requisiti di **Scalabilità**: il server può scalare verticalmente o orizzontalmente indipendentemente dal numero di dashboard connesse, elaborando dozzine di *request* concorrenti senza bloccare l'interfaccia utente.

---

## 🧠 2. Il Pattern Multi-Agente (Hub & Spoke)

La logica decisionale risiede nel file `multi_agent.py` (e nei componenti definiti nel grafo LangGraph), che instanzia un grafo ciclico (`StateGraph`). Al centro di questo grafo siede l'Agente **Supervisor** (l'Hub), circondato dai nodi operativi (gli Spoke: `Decomposer`, `Retriever`, `Reasoner`, `Veracity`).

```text
                                [API REST Endpoint]
                                         │
                                         ▼
               ┌────────────────► [SUPERVISOR] ◄────────────────┐
               │                 (Qwen2.5 LLM)                  │
               │                         │                      │
               ▼                         ▼                      ▼
         [Decomposer]               [Retriever]            [Reasoner / Veracity]
      (Tool Calling JSON)       (Ramo KB / Ramo LIT)     (Map-Reduce / Cross-Encoder)
```

### Il Ruolo dell'Orchestratore (Il Sistema nel suo Complesso)
L'**Orchestratore** è il macro-componente che incapsula le API FastAPI, le connessioni ai database (MongoDB, FAISS) e l'istanza del grafo LangGraph. Si occupa di:
- Ricevere le richieste di fact-checking dal client e validare gli input (es. testo del claim o eventuali allegati).
- Gestire la memoria a lungo e breve termine, e la persistenza dello stato delle conversazioni (tramite `langgraph-checkpoint-mongodb`).
- Fornire e allocare le risorse hardware necessarie agli agenti (modelli AI in esecuzione locale o remota).
- Raccogliere e confezionare il responso finale formattato da restituire al client.

### Il Ruolo del Supervisore (Il "Cervello" del Grafo)
Il **Supervisor** è un agente primario, mosso da un LLM avanzato (come Qwen2.5), specializzato nel ragionamento orchestrato e nel routing dinamico. Funziona letteralmente come il direttore d'orchestra all'interno del processo di verifica:
1. **Analisi Iniziale e Delega**: Non appena il claim entra nel grafo, viene valutato dal Supervisor. Questo decide quale "specialista" (Spoke) chiamare in base al task da compiere. Ad esempio, stabilisce se un claim è sufficientemente semplice da passare dritto al `Retriever`, oppure se necessita di essere diviso in più parti dal `Decomposer`.
2. **Controllo dello Stato**: Al termine del lavoro di ciascun nodo operativo, il flusso ritorna obbligatoriamente al Supervisor. Esso valuta l'output ricevuto dallo specialista e decide la prossima mossa (es. i documenti recuperati bastano? Se sì, vai al `Reasoner`, se no, esegui nuovamente il `Retriever` usando parametri diversi).
3. **Terminazione**: Una volta che l'intero workflow porta a un risultato solido (giudizio calcolato sul claim), il Supervisor dichiara chiusa l'esecuzione, facendo in modo che lo stato finale esca dal grafo.

### I Nodi Operativi (Spoke)
- **Decomposer**: Si attiva per analizzare claim molto complessi o formati da più affermazioni congiunte. Le suddivide in *sub-claims* atomici e testabili singolarmente, facilitando il reperimento delle fonti. Produce generalmente un output strutturato (JSON) invocando dei *Tool*.
- **Retriever**: L'agente di ricerca delle evidenze scientifiche/mediche a supporto o smentita del claim. In MedFactCheck può operare lungo due direttrici:
  - **Ramo KB**: Interroga Knowledge Base interne altamente strutturate sfruttando le potenzialità della Retrieval-Augmented Generation (RAG) mediante indici vettoriali (es. FAISS) o classici (BM25).
  - **Ramo LIT**: Raccoglie evidenze dalla letteratura e database esterni interrogabili se i documenti locali non sono sufficienti.
- **Reasoner**: Riceve in input le prove fornite dal Retriever associate al claim. Attraverso pattern come *Map-Reduce*, ragiona sui dati testuali, filtrando le informazioni rumorose, per generare una sintesi delle evidenze mirata e unificata.
- **Veracity**: Modulo finale di classificazione e allineamento (spesso basato su un modello Cross-Encoder dedicato, es. DeBERTa). Prende in esame l'accoppiamento semantico tra `[Claim]` e `[Sintesi delle Evidenze]` restituendo una *label* di veridicità (es. *Supported*, *Refuted*, *Not Enough Info*) e il relativo punteggio di confidenza matematica.

---

## 🚀 3. Flusso Tipico di Esecuzione (Workflow)

Per comprendere meglio come i componenti collaborino in sincrono, ecco il ciclo di vita standard di una richiesta:

1. **Richiesta Iniziale**: L'utente inserisce un'affermazione sospetta sulla dashboard (es. "L'aspirina cura definitivamente l'influenza"), che invia una POST all'Orchestratore (API).
2. **Presa in Carico**: L'Orchestratore fa partire la macchina LangGraph. Il claim viene inserito nello stato condiviso.
3. **Routing (Supervisor)**: Il Supervisor ispeziona il claim. Essendo un claim unitario, salta il Decomposer e comanda al Retriever di cercare prove al riguardo.
4. **Recupero (Retriever)**: Il Retriever esegue query sui database vettoriali/letteratura, estrae 5 documenti medici o *abstracts* pertinenti e cede di nuovo il turno al Supervisor.
5. **Ragionamento (Supervisor -> Reasoner)**: Il Supervisor verifica che i 5 documenti siano stati ottenuti, e demanda al Reasoner il compito di leggerli per estrarne il significato profondo, mettendoli in relazione col claim.
6. **Giudizio (Veracity)**: Il Reasoner condensa le informazioni; il blocco Veracity applica il modello Cross-Encoder calcolando la sentenza finale: "Refuted" (Confidenza: 99.2%).
7. **Risposta e Salvataggio (Orchestratore)**: Il Supervisor riconosce che il workflow è terminato. L'Orchestratore impacchetta i risultati, salva permanentemente tutto in MongoDB e risponde con successo alla Dashboard, che mostra a schermo i dati.