"""
Esempio Architetturale: Sistema Multi-Agente (Pattern Supervisor) per MedFactCheck
"""
from typing import Annotated, Any, Dict, Sequence
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import asyncio
import os
import sys

# Aggiunge la directory 'src' al path per consentire importazioni da agent.utils
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from agent.utils.Qwen import QwenNF4
from agent.utils.Retriever import MedFactCheckRetriever
from agent.utils.DeBERTa import DebertaVeracityNode
from agent.utils.Storage import StorageManager

# Import dei Nodi Worker separati per mantenere l'architettura modulare
from agent.utils.ClaimIngestionAndDecompositionFunc import IngestionAndDecomposerAgent
from agent.utils.RetrievalFunc import RetrieverAgent
from agent.utils.ReasoningFunc import ReasonerAgent
from agent.utils.VeracityFunc import VeracityAgent

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

# Inizializzazione globale dei modelli (fuori dallo stato per permettere la serializzazione)
print("⏳ Inizializzazione globale dei modelli in corso.")
qwen_instance = QwenNF4()
retriever_instance = MedFactCheckRetriever()
deberta_instance = DebertaVeracityNode()
storage_instance = StorageManager()


# 2. LO STATO MULTI-AGENTE
class MultiAgentState(TypedDict):
    # I messaggi sono fondamentali nei sistemi multi-agente per la "memoria" condivisa
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Lo stato mantiene le variabili necessarie, ma sarà il supervisore a guidare
    claim_input: dict
    claim_id: str  # Utilizzato dal DB Manager per la tracciabilità
    sub_claims: list
    routing_info: dict
    retrieved_docs: dict
    reasoning_output: dict
    veracity_results: dict
    next_agent: str  # Indica quale agente deve agire nel prossimo step


# 3. IL NODO SUPERVISORE
class SupervisorAgent:
    def __init__(self, decomposition_model: Any, storage_instance: Any):
        self.qwen_model = decomposition_model
        self.storage_instance = storage_instance

    def _sync_to_db(self, state: MultiAgentState) -> str:
        """Sincronizzazione Python pura (upsert) in Mongo."""
        claim_text = state.get("claim_input", {}).get("text", "")
        claim_id = state.get("claim_id")
        
        if not claim_id and claim_text:
            claim_id = self.storage_instance.save_claim(original_text=claim_text)
            
            # Salvataggio dell'immagine in MongoDB se presente
            image_b64 = state.get("claim_input", {}).get("image_b64")
            if claim_id and image_b64:
                try:
                    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
                    db = client[os.getenv("MONGO_DB_NAME", "medfactcheck")]
                    db["final_results"].update_one({"claim_id": claim_id}, {"$set": {"image_b64": image_b64}})
                except Exception as e:
                    print(f"[SUPERVISORE] Errore salvataggio immagine: {e}")
                    
        if not claim_id:
            return ""
            
        sub_claims = state.get("sub_claims", [])
        routing_info = state.get("routing_info", {})
        if sub_claims:
            res_dict = {
                "sub_claims": [{"claim": sc, "routes": routing_info.get(sc, [])} for sc in sub_claims],
            }
            self.storage_instance.save_claim_decomposition(claim_id, res_dict)
            
        retrieved_docs = state.get("retrieved_docs", {})
        if retrieved_docs:
            for sc, docs in retrieved_docs.items():
                self.storage_instance.save_evidence(claim_id, {"claim": sc, "evidence_passages": docs})
            self.storage_instance.update_claim_status(claim_id, "retrieved")
            
        veracity_results = state.get("veracity_results", {})
        if veracity_results:
            for sc, veracity_output in veracity_results.items():
                self.storage_instance.save_verdict(claim_id, veracity_output)
                
        return claim_id

    def __call__(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Il Supervisore legge 'state["messages"]' e decide:
        - Manca la decomposizione? -> "Decomposer"
        - Ci sono sub-claims ma mancano evidenze? -> "Retriever"
        - È fallito il retrieval? -> Chiede al "Decomposer" di riprovare.
        - Tutto finito? -> "FINISH"
        """
        print("\n[SUPERVISORE] Analizzo lo stato e decido la prossima mossa...")
        
        # Sincronizzazione centralizzata: il codice Python salva tutto prima del LLM
        claim_id = self._sync_to_db(state)
        
        messages_history = ""
        for msg in state.get("messages", []):
            messages_history += f"[{msg.name}]: {msg.content}\n"
            
        if not messages_history:
            messages_history = "Nessuna operazione finora. Il claim è appena arrivato."
            
        try:
            raw_response = self.qwen_model.supervise(messages_history)
            scelta = raw_response.strip().replace(".", "").replace('"', '')
            
            valid_agents = ["Decomposer", "Retriever", "Reasoner", "Veracity", "FINISH"]
            if scelta not in valid_agents:
                print(f"[SUPERVISORE] Output non valido riconosciuto: '{scelta}'. Forzo la fine.")
                scelta = "FINISH"
        except Exception as e:
            print(f"[SUPERVISORE] Errore nell'inferenza, termino. Dettagli: {e}")
            scelta = "FINISH"
            
        log_msg = f"Valutazione dello stato completata. Il prossimo task è assegnato a: '{scelta}'." if scelta != "FINISH" else "Elaborazione completata. Flusso terminato."
            
        if scelta == "FINISH" and claim_id:
            agent_trace = [f"[{m.name}]: {m.content}" for m in state.get("messages", []) if hasattr(m, 'name')]
            agent_trace.append(f"[Supervisor]: {log_msg}")
            self.storage_instance.aggregate_final_verdict(claim_id, agent_trace=agent_trace)
            
        print(f"[SUPERVISORE] Smisto il task a: {scelta}")
        return {
            "next_agent": scelta, 
            "claim_id": claim_id,
            "messages": [AIMessage(content=log_msg, name="Supervisor")]
        }


# 5. COSTRUZIONE DEL GRAFO A STELLA (HUB & SPOKE)
builder = StateGraph(MultiAgentState)

# Istanziazione OOP degli agenti con Dependency Injection
supervisor_agent = SupervisorAgent(qwen_instance, storage_instance)
decomposer_agent = IngestionAndDecomposerAgent(qwen_instance)
retriever_agent = RetrieverAgent(qwen_instance, retriever_instance)
reasoner_agent = ReasonerAgent(qwen_instance)
veracity_agent = VeracityAgent(deberta_instance)

# Aggiunta Nodi
builder.add_node("Supervisor", supervisor_agent)
builder.add_node("Decomposer", decomposer_agent)
builder.add_node("Retriever", retriever_agent)
builder.add_node("Reasoner", reasoner_agent)
builder.add_node("Veracity", veracity_agent)

# Tutti i Worker operativi restituiscono SEMPRE il controllo al Supervisore
builder.add_edge("Decomposer", "Supervisor")
builder.add_edge("Retriever", "Supervisor")
builder.add_edge("Reasoner", "Supervisor")
builder.add_edge("Veracity", "Supervisor")

# Il Supervisore decide in modo condizionale a chi inviare il flusso
builder.add_conditional_edges(
    "Supervisor",
    lambda state: state["next_agent"], # Legge la stringa outputtata dall'LLM
    {
        "Decomposer": "Decomposer",
        "Retriever": "Retriever",
        "Reasoner": "Reasoner",
        "Veracity": "Veracity",
        "FINISH": END
    }
)

# L'entrypoint è sempre il Supervisore
builder.set_entry_point("Supervisor")

# Abilitazione del Checkpointer LangGraph per MongoDB (Tracking dei Thread)
client = MongoClient("mongodb://localhost:27017/")
memory_saver = MongoDBSaver(client)

multi_agent_graph = builder.compile(checkpointer=memory_saver)

async def main():
    print("🚀 Avvio Sistema Multi-Agente MedFactCheck...")
    storage_instance.ensure_indexes()
    import uuid
    
    while True:
        user_input = input("\nInserisci un claim medico da verificare (o 'exit' per uscire): ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue
            
        inputs = {
            "claim_input": {"text": user_input},
            "claim_id": "",
            "sub_claims": [],
            "routing_info": {},
            "retrieved_docs": {},
            "reasoning_output": {},
            "veracity_results": {},
            "messages": []
        }
        
        # LangGraph richiede un thread_id quando si usa un Checkpointer
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        print("\n🔄 Elaborazione Multi-Agente in corso...\n")
        async for output in multi_agent_graph.astream(inputs, config=config, stream_mode="updates"):
            pass
            
        print("✅ Elaborazione Completata!")

if __name__ == "__main__":
    asyncio.run(main())