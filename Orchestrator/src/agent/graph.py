"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph
from typing_extensions import NotRequired, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

# Questo è un workaround per far funzionare correttamente le importazioni quando langgraph esegue il file.
# Aggiunge la directory 'src' al percorso di Python, consentendo importazioni assolute da 'agent'.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from agent.utils.Qwen import *
from agent.utils.DeBERTa import *
from agent.utils.ClaimDecompositionFunc import *
from agent.utils.ReasoningVeracityFunc import *
from agent.utils.RetrievalFunc import *
from agent.utils.Storage import *

class State(TypedDict):
    """Input state for the agent.

    Defines the structure of data that flows through the graph.
    Using TypedDict is a common practice for defining state in LangGraph.
    """

    # Input claim from the user
    claim_input: dict 
    claim_id: NotRequired[str]
    
    # Output from decomposition node
    decomposition_input: NotRequired[Dict[str, Any ]]
    sub_claims: NotRequired[List[str]]
    routing_info: NotRequired[Dict[str, List[str]]]
    decomposition_model: NotRequired[str]
    decomposer_reasoning: NotRequired[str]

    # Output from input_check node
    input_checked: NotRequired[bool]
    json_checked: NotRequired[bool]


    # Output from retrieval node
    retrieved_docs: NotRequired[Dict[str, str]]

    # Output from reasoning node
    reasoning_output: NotRequired[str]
    reasoning_model: NotRequired[str]
    retrieval_models: NotRequired[str]

    # Final output from veracity node
    veracity: NotRequired[str]
    veracity_model: NotRequired[str]
    
    # Output from db init
    db_initialized: NotRequired[bool]

def check_init_branch(state: State) -> str:
    """Determina se l'inizializzazione di tutti i modelli ha avuto successo."""
    required_checks = [
        "decomposition_model",
        "retrieval_models",
        "reasoning_model",
        "veracity_model",
        "db_initialized"
    ]
    if all(state.get(check) for check in required_checks):
        print("-> Inizializzazione di tutti i modelli e DB completata con successo.")
        return "continue"
    else:
        missing = [check for check in required_checks if not state.get(check)]
        print(f"❌ Errore: Inizializzazione fallita per i seguenti componenti: {missing}. Termino il flusso.")
        return "end"

def check_json_branch(state: State) -> str:
    """Determina se la preparazione dell'input ha avuto successo."""
    if state.get("json_checked"):
        print("-> Input preparato correttamente, procedo con la decomposizione.")
        return "continue"
    print("-> Errore nella preparazione dell'input, termino il flusso.")
    return "end"

def check_decomposition_branch(state: State) -> str:
    """Determina se la decomposizione ha prodotto risultati validi."""
    if state.get("input_checked"):
        print("-> Decomposizione valida, procedo con il recupero delle evidenze.")
        return "continue"
    print("-> Decomposizione fallita o non valida, termino il flusso.")
    return "end"

def check_save_decomposition_branch(state: State) -> str:
    """Determina se il salvataggio della decomposizione ha avuto successo."""
    if state.get("claim_id"):
        print("-> Salvataggio completato, procedo con il recupero delle evidenze.")
        return "continue"
    print("-> Errore nel salvataggio, termino il flusso.")
    return "end"

async def init_db(state: State) -> Dict[str, Any]:
    """Inizializza e verifica la connessione al database MongoDB."""
    print("--- INIT DATABASE ---")
    try:
        storage = StorageManager()
        storage.close()
        print("✅ Connessione al database MongoDB verificata con successo.")
        return {"db_initialized": True}
    except Exception as e:
        print(f"❌ Errore di connessione al database MongoDB: {e}")
        return {"db_initialized": False}


async def save_decomposition_node(state: State) -> Dict[str, Any]:
    """Salva i risultati della decomposizione nel database MongoDB usando StorageManager."""
    print("--- SAVING DECOMPOSITION TO STORAGE ---")
    try:
        # Istanzia lo StorageManager (la connessione viene gestita in automatico)
        storage = StorageManager()
        
        sub_claims_list = state.get("sub_claims", [])
        routing_info = state.get("routing_info", {})
        
        # Ricostruisce il dizionario nel formato atteso dallo StorageManager
        decomposer_output = {
            "reasoning": state.get("decomposer_reasoning", ""),
            "sub_claims": [{"claim": sc, "routes": routing_info.get(sc, [])} for sc in sub_claims_list]
        }
        
        original_text = state.get("claim_input", {}).get("text", "")
        
        # Salva a database
        claim_id = storage.save_claim_decomposition(
            original_text=original_text,
            decomposer_output=decomposer_output,
            source_type="text"
        )
        storage.close()
        return {"claim_id": claim_id}
    except Exception as e:
        print(f"❌ Errore durante il salvataggio della decomposizione: {e}")
        return {}

async def retrieve_evidence(state: State) -> Dict[str, Any]:
    """Retrieves evidence for sub-claims based on the routing info."""
    # model_info = await init_retrieval(state) # Rimosso, ora è un nodo separato
    print(f"--- RETRIEVING EVIDENCE (using {state.get('retrieval_models', 'Unknown')}) ---")
    # Placeholder logic: Here you would implement the retrieval from KB (BM25) and LIT (BM25 + BioBERT).
    retrieved_docs = {
        "sub_claim_1": "Evidence from BM25 on local KB.",
        "sub_claim_2": "Evidence from BM25 (KB) and BioBERT (LIT).",
    }
    print("Retrieved evidence for sub-claims.")
    return {"retrieved_docs": retrieved_docs}


async def reason_on_evidence(state: State) -> Dict[str, Any]:
    """Reasons over the retrieved evidence to evaluate the sub-claims."""
    print(f"--- REASONING ON EVIDENCE (using {state.get('reasoning_model', 'Unknown')}) ---")
    # Placeholder logic: Call the reasoning model (e.g., Qwen).
    reasoning_output = "Reasoning complete. The evidence appears to support the claims."
    print(reasoning_output)
    return {"reasoning_output": reasoning_output}


async def determine_veracity(state: State) -> Dict[str, Any]:
    """Determines the final veracity of the main claim."""
    # model_info = await init_veracity(state) # Rimosso, ora è un nodo separato
    print(f"--- DETERMINING VERACITY (using {state.get('veracity_model', 'Unknown')}) ---")
    # Placeholder logic: Call the veracity model (e.g., BioBERTa).
    veracity = "Fact-check result: Largely True"
    print(veracity)
    return {"veracity": veracity}

async def join_init(state: State) -> None:
    """Nodo di giunzione dopo l'inizializzazione parallela."""
    print("--- Inizializzazione parallela completata, avvio controllo. ---")
    return None


# Define the graph
workflow = StateGraph(State)

# Aggiungi i nodi che verranno eseguiti in parallelo
workflow.add_node("init_qwen_istance", init_qwen_istance)
workflow.add_node("init_retrieval", init_retrieval)
workflow.add_node("init_deberta_istance", init_deberta_istance)
workflow.add_node("init_veracity", init_veracity)
workflow.add_node("init_db", init_db)

workflow.add_node("join_init", join_init)
workflow.add_node("input_to_json", input_to_json)
workflow.add_node("decompose_subclaims_check", decompose_subclaims_check)
workflow.add_node("save_decomposition", save_decomposition_node)


# Aggiungi i nodi di elaborazione principali
workflow.add_node("decompose", run_decomposition)
workflow.add_node("retrieve", retrieve_evidence)
workflow.add_node("reason", reason_on_evidence)
workflow.add_node("veracity", determine_veracity)

# 1. Esegui i nodi di inizializzazione in parallelo partendo da __start__
workflow.add_edge("__start__", "init_qwen_istance")
workflow.add_edge("__start__", "init_retrieval")
workflow.add_edge("__start__", "init_veracity")
workflow.add_edge("__start__", "init_deberta_istance")
workflow.add_edge("__start__", "init_db")

# 2. Ricongiungi i rami paralleli al nodo `join_init`
workflow.add_edge(["init_qwen_istance", "init_retrieval", "init_deberta_istance", "init_veracity", "init_db"], "join_init")

# 3. Aggiungi il branch condizionale dopo l'inizializzazione
workflow.add_conditional_edges(
    "join_init",
    check_init_branch,
    {"continue": "input_to_json", "end": "__end__"}
)

workflow.add_conditional_edges(
    "input_to_json",
    check_json_branch,
    {"continue": "decompose", "end": "__end__"}
)

workflow.add_edge("decompose", "decompose_subclaims_check")

workflow.add_conditional_edges(
    "decompose_subclaims_check",
    check_decomposition_branch,
    {"continue": "save_decomposition", "end": "__end__"}
)

# 5. Continua con il flusso di fact-checking
workflow.add_conditional_edges(
    "save_decomposition",
    check_save_decomposition_branch,
    {"continue": "retrieve", "end": "__end__"}
)
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "veracity")
workflow.add_edge("veracity", "__end__")

graph = workflow.compile(name="Medical Fact-Checking Graph")
