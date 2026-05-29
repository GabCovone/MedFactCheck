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

2. (Optional) Customize the code and project as needed. Create a `.env` file if you need to use secrets.

```bash
cp .env.example .env
```

If you want to enable LangSmith tracing, add your LangSmith API key to the `.env` file.

```text
# .env
LANGSMITH_API_KEY=lsv2...
```

3. Start the LangGraph Server.

```shell
langgraph dev
```

For more information on getting started with LangGraph Server, [see here](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/).

## How to customize

1. **Define runtime context**: Modify the `Context` class in the `graph.py` file to expose the arguments you want to configure per assistant. For example, in a chatbot application you may want to define a dynamic system prompt or LLM to use. For more information on runtime context in LangGraph, [see here](https://langchain-ai.github.io/langgraph/agents/context/?h=context#static-runtime-context).

2. **Extend the graph**: The core logic of the application is defined in [graph.py](./src/agent/graph.py). You can modify this file to add new nodes, edges, or change the flow of information.

## Development

While iterating on your graph in LangGraph Studio, you can edit past state and rerun your app from previous states to debug specific nodes. Local changes will be automatically applied via hot reload.

Follow-up requests extend the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

For more advanced features and examples, refer to the [LangGraph documentation](https://langchain-ai.github.io/langgraph/). These resources can help you adapt this template for your specific use case and build more sophisticated conversational agents.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates, allowing you to analyze and optimize your chatbot's performance.
