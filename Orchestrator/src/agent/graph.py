"""MedFactCheck Main Graph Orchestrator.

Defines the state graph and nodes for claim decomposition, evidence retrieval,
reasoning generation, and final veracity classification.
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
from agent.utils.ReasoningFunc import *
from agent.utils.VeracityFunc import *
from agent.utils.RetrievalFunc import *
from agent.utils.Storage import *
from agent.utils.Retriever import *

class State(TypedDict):
    """Input state for the agent.

    Defines the structure of data that flows through the graph.
    Using TypedDict is a common practice for defining state in LangGraph.
    """

    # Input claim from the user
    claim_input: dict 
    claim_id: NotRequired[str]
    claim_saved: NotRequired[bool]
    
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
    retrieved_docs: NotRequired[Dict[str, Any]]
    evidence_saved: NotRequired[bool]
    retrieval_output_checked: NotRequired[bool]

    # Output from reasoning node
    reasoning_output: NotRequired[Dict[str, str]]
    reasoning_checked: NotRequired[bool]
    reasoning_model: NotRequired[str]
    retrieval_models: NotRequired[str]

    # Final output from veracity node
    veracity_results: NotRequired[Dict[str, dict]]
    veracity_checked: NotRequired[bool]
    verdicts_saved: NotRequired[bool]
    final_result: NotRequired[Dict[str, Any]]
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

def check_save_claim_branch(state: State) -> str:
    """Determina se il salvataggio iniziale del claim ha avuto successo."""
    if state.get("claim_saved"):
        print("-> Claim iniziale salvato, procedo con la validazione dell'input.")
        return "continue"
    print("-> Errore nel salvataggio iniziale del claim, termino il flusso.")
    return "end"

async def save_claim_node(state: State) -> Dict[str, Any]:
    """Salva il claim originale all'inizio del flusso."""
    print("--- SAVING INITIAL CLAIM TO STORAGE ---")
    try:
        storage = StorageManager()
        claim_input = state.get("claim_input")
        
        if not claim_input or not isinstance(claim_input, dict):
            print("❌ Input non valido per il salvataggio.")
            return {"claim_saved": False}
            
        original_text = claim_input.get("text", "")
        
        source_type = "text"
        if claim_input.get("image"):
            source_type = "image"
        
        # Inizializziamo il record per generare il claim_id e lo status received
        claim_id = storage.save_claim(
            original_text=original_text,
            source_type=source_type
        )
        
        storage.close()
        return {"claim_id": claim_id, "claim_saved": True}
    except Exception as e:
        print(f"❌ Errore durante il salvataggio iniziale del claim: {e}")
        return {"claim_saved": False}

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
        
        claim_id = state.get("claim_id")
        
        # Aggiorna il record a database
        storage.save_claim_decomposition(
            claim_id=claim_id,
            decomposer_output=decomposer_output
        )
        storage.close()
        return {"claim_id": claim_id}
    except Exception as e:
        print(f"❌ Errore durante il salvataggio della decomposizione: {e}")
        return {}

def check_save_evidence_branch(state: State) -> str:
    """Determina se il salvataggio delle evidenze ha avuto successo."""
    if state.get("evidence_saved"):
        print("-> Evidenze salvate, procedo con il reasoning.")
        return "continue"
    print("-> Errore nel salvataggio delle evidenze, termino il flusso.")
    return "end"

async def save_evidence_node(state: State) -> Dict[str, Any]:
    """Salva le evidenze recuperate nel database MongoDB usando StorageManager."""
    print("--- SAVING EVIDENCE TO STORAGE ---")
    try:
        storage = StorageManager()
        claim_id = state.get("claim_id")
        retrieved_docs = state.get("retrieved_docs", {})
        
        for sc, docs in retrieved_docs.items():
            retriever_output = {
                "claim_id": claim_id,
                "claim": sc,
                "evidence_passages": docs
            }
            storage.save_evidence(claim_id=claim_id, retriever_output=retriever_output)
        
        storage.update_claim_status(claim_id, "retrieved")
        storage.close()
        return {"evidence_saved": True}
    except Exception as e:
        print(f"❌ Errore durante il salvataggio delle evidenze: {e}")
        return {"evidence_saved": False}

def check_retrieval_output_branch(state: State) -> str:
    """Determina se l'output del retrieval è valido."""
    if state.get("retrieval_output_checked"):
        print("-> Output del retrieval valido, procedo con il salvataggio delle evidenze.")
        return "continue"
    print("-> Errore nell'output del retrieval, termino il flusso.")
    return "end"

def check_reasoning_branch(state: State) -> str:
    """Determina se il ragionamento ha prodotto risultati validi."""
    if state.get("reasoning_checked"):
        print("-> Ragionamento valido, procedo con la classificazione della veridicità.")
        return "continue"
    print("-> Errore nel ragionamento, termino il flusso.")
    return "end"

def check_veracity_branch(state: State) -> str:
    """Determina se la classificazione della veridicità ha prodotto risultati validi."""
    if state.get("veracity_checked"):
        print("-> Veridicità valutata, procedo con il salvataggio dei verdetti.")
        return "continue"
    print("-> Errore nella valutazione della veridicità, termino il flusso.")
    return "end"

async def save_verdicts_node(state: State) -> Dict[str, Any]:
    """Salva i verdetti di tutti i sub-claims e aggrega il verdetto finale nel database."""
    print("--- SAVING VERDICTS TO STORAGE ---")
    try:
        storage = StorageManager()
        claim_id = state.get("claim_id")
        veracity_results = state.get("veracity_results", {})
        
        for sc, veracity_output in veracity_results.items():
            storage.save_verdict(claim_id=claim_id, veracity_output=veracity_output)
        
        # Esegue anche l'aggregazione finale richiesta dal Coordinator Agent
        final_doc = storage.aggregate_final_verdict(claim_id=claim_id, agent_trace=[])
        
        storage.close()
        print("✅ Tutti i verdetti sono stati salvati e il claim è completato.")
        return {"verdicts_saved": True, "final_result": final_doc}
    except Exception as e:
        print(f"❌ Errore durante il salvataggio dei verdetti: {e}")
        return {"verdicts_saved": False}

def check_save_verdicts_branch(state: State) -> str:
    """Determina se il salvataggio dei verdetti ha avuto successo."""
    if state.get("verdicts_saved"):
        return "continue"
    print("-> Errore nel salvataggio dei verdetti, termino il flusso.")
    return "end"

async def print_final_result_node(state: State) -> Dict[str, Any]:
    """Stampa a schermo il riepilogo finale del fact-checking."""
    final_result = state.get("final_result", {})
    if not final_result:
        print("⚠️ Nessun risultato finale disponibile per la stampa.")
        return {}
        
    print("\n" + "="*60)
    print("🏆 RISULTATO FINALE DEL FACT-CHECKING 🏆".center(60))
    print("="*60)
    print(f"🔹 CLAIM: {final_result.get('original_text')}")
    print(f"🔹 VERDETTO: {final_result.get('final_verdict')}")
    print(f"🔹 CONFIDENZA MEDIA: {final_result.get('avg_confidence', 0.0):.2%}")
    print("\n🔹 DETTAGLIO SUB-CLAIMS:")
    for sc in final_result.get('sub_verdicts', []):
        print(f"   [{sc.get('verdict')}] (Conf: {sc.get('confidence_score', 0.0):.2%}) -> {sc.get('sub_claim')}")
    print("="*60 + "\n")
    return {}

async def join_init(state: State) -> None:
    """Nodo di giunzione dopo l'inizializzazione parallela."""
    print("--- Inizializzazione parallela completata, avvio controllo. ---")
    return None


# Define the graph
workflow = StateGraph(State)

# Aggiungi i nodi che verranno eseguiti in parallelo
workflow.add_node("init_qwen_instance", init_qwen_instance)
workflow.add_node("init_retrieval", init_retrieval)
workflow.add_node("init_deberta_instance", init_deberta_instance)
workflow.add_node("init_db", init_db)

workflow.add_node("join_init", join_init)
workflow.add_node("save_claim", save_claim_node)
workflow.add_node("input_to_json", input_to_json)
workflow.add_node("decompose_subclaims_check", decompose_subclaims_check)
workflow.add_node("save_decomposition", save_decomposition_node)

workflow.add_node("save_evidence", save_evidence_node)

# Aggiungi i nodi di elaborazione principali
workflow.add_node("decompose", run_decomposition)
workflow.add_node("retrieve", retrieve_evidence)
workflow.add_node("retrieval_output_check", retrieval_output_check)
workflow.add_node("reason", run_reasoning)
workflow.add_node("reasoning_output_check", reasoning_output_check)
workflow.add_node("veracity", run_veracity)
workflow.add_node("veracity_output_check", veracity_output_check)
workflow.add_node("save_verdicts", save_verdicts_node)
workflow.add_node("print_final_result", print_final_result_node)

# 1. Esegui i nodi di inizializzazione in parallelo partendo da __start__
workflow.add_edge("__start__", "init_qwen_instance")
workflow.add_edge("__start__", "init_retrieval")
workflow.add_edge("__start__", "init_deberta_instance")
workflow.add_edge("__start__", "init_db")

# 2. Ricongiungi i rami paralleli al nodo `join_init`
workflow.add_edge(["init_qwen_instance", "init_retrieval", "init_deberta_instance", "init_db"], "join_init")

# 3. Aggiungi il branch condizionale dopo l'inizializzazione
workflow.add_conditional_edges(
    "join_init",
    check_init_branch,
    {"continue": "save_claim", "end": "__end__"}
)

workflow.add_conditional_edges(
    "save_claim",
    check_save_claim_branch,
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

workflow.add_edge("retrieve", "retrieval_output_check")

workflow.add_conditional_edges(
    "retrieval_output_check",
    check_retrieval_output_branch,
    {"continue": "save_evidence", "end": "__end__"}
)

workflow.add_conditional_edges(
    "save_evidence",
    check_save_evidence_branch,
    {"continue": "reason", "end": "__end__"}
)

workflow.add_edge("reason", "reasoning_output_check")

workflow.add_conditional_edges(
    "reasoning_output_check",
    check_reasoning_branch,
    {"continue": "veracity", "end": "__end__"}
)

workflow.add_edge("veracity", "veracity_output_check")

workflow.add_conditional_edges(
    "veracity_output_check",
    check_veracity_branch,
    {"continue": "save_verdicts", "end": "__end__"}
)

workflow.add_conditional_edges(
    "save_verdicts",
    check_save_verdicts_branch,
    {"continue": "print_final_result", "end": "__end__"}
)

workflow.add_edge("print_final_result", "__end__")

graph = workflow.compile(name="Medical Fact-Checking Graph")
