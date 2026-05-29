# 🖥️ MedFactCheck - Dashboard Client

Il modulo **Dashboard** costituisce l'interfaccia utente (Front-End) del sistema **MedFactCheck**. È un'applicazione web ultra-leggera e interattiva sviluppata in **Streamlit**, progettata per offrire un'esperienza utente reattiva e trasparente durante l'intero processo di fact-checking di claim biomedici.

---

## 🎯 Obiettivi e Responsabilità

In linea con l'architettura disaccoppiata del sistema, la Dashboard ha il compito esclusivo di gestire la presentazione dei dati e l'interazione con l'utente, delegando interamente il carico computazionale pesante (modelli IA, calcolo vettoriale, inferenza LLM) al server di Back-End (Orchestratore).

Le sue responsabilità principali includono:
1. **Acquisizione dell'Input (Multimodale)**: Raccogliere affermazioni sospette testuali (claim) e/o file multimediali (immagini biomediche, referti) inserite dall'utente, permettendo analisi puramente visive, puramente testuali o ibride.
2. **Delega al Back-End**: Inviare asincronamente il payload al server FastAPI (endpoint `/verify`) tramite chiamate HTTP POST (`requests.post`).
3. **Monitoraggio e Visualizzazione**: Interrogare localmente **MongoDB** per estrarre e mostrare i log e le decisioni prese dal Supervisore e dai nodi operativi (Decomposer, Retriever, Reasoner, Veracity).
4. **Rendering dei Risultati**: Visualizzare le metriche di output, la scomposizione dei sub-claims, i documenti recuperati (evidenze mediche) e la sentenza finale (es. *Supported*, *Refuted*, *Not Enough Info*) accompagnata dal punteggio di confidenza.
5. **Ricerca e Filtri Avanzati**: Esplorare l'intero database storico di MongoDB filtrando per parola chiave (Regex testuale), Verdetto, Fonte bibliografica (es. Europe PMC, DisGeNET) e Finestra Temporale (es. Ultime 24h, 7 giorni).
6. **Reportistica PDF**: Generare ed esportare dinamicamente un report PDF strutturato per ogni claim verificato, grazie all'integrazione con la libreria `reportlab`.

---

## 🏗️ I Vantaggi del Disaccoppiamento

Mantenere la Dashboard Streamlit come client separato assicura enormi vantaggi in termini di **Scalabilità**:
- L'interfaccia occupa risorse di calcolo minime e garantisce un'alta responsività della UI.
- La VRAM necessaria per i modelli (Qwen2.5, DeBERTa, indici FAISS) non è intaccata né frammentata da operazioni di UI.
- Decine di utenti possono avviare e usare la Dashboard contemporaneamente, sfruttando le code asincrone dell'Orchestratore, senza il rischio di bloccare il rendering a schermo.

---

## ⚡ Strategia di Caching (Smart Keying)

Per prevenire il sovraccarico del database e garantire un'esperienza utente istantanea, la Dashboard implementa una rigorosa politica di **Caching Parametrico**:
- Le query effettuate a MongoDB vengono memorizzate in RAM (tramite `@st.cache_data`) con un **Time-to-Live (TTL) di 1 ora**.
- I risultati sono legati alla specifica combinazione di filtri richiesti dall'utente. Questo significa che navigare tra le varie schermate di un claim (Verdetto, Evidenze, Log) non scaturisce nessuna nuova query al DB, servendo i dati in pochi millisecondi.
- **Invalidazione Automatica**: Al momento della sottomissione di un *nuovo* claim verso l'API, la cache viene resettata dinamicamente (`st.cache_data.clear()`), garantendo che l'interfaccia mostri immediatamente il nuovo dato processato a tutti gli utenti attivi.

---

## 🚀 Avvio Rapido

Prima di lanciare la Dashboard, assicurati che i servizi di base siano attivi (es. servizio MongoDB e l'Orchestratore `uvicorn`). Per avviare l'interfaccia, esegui:

```bash
streamlit run Dashboard.py
```
👉 **http://localhost:8501**