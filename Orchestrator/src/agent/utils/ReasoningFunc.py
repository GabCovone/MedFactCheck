from typing import Dict, Any

async def reasoning_input_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Controlla che i dati necessari per il reasoning (sub_claims e documenti) siano presenti."""
    print("--- CHECKING REASONING INPUT ---")
    sub_claims = state.get("sub_claims")
    retrieved_docs = state.get("retrieved_docs")
    
    if sub_claims and isinstance(sub_claims, list) and isinstance(retrieved_docs, dict):
        print(f"✅ Check superato: Input valido per {len(sub_claims)} sub-claims.")
        return {"reasoning_input_checked": True}
    else:
        print("❌ Check fallito: Input per il reasoning mancante o non valido.")
        return {"reasoning_input_checked": False}

async def run_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue il modello di ragionamento sui sub-claims e i documenti recuperati
    per generare una Chain-of-Thought (CoT).
    """
    print("--- RUNNING REASONING ON EVIDENCE ---")
    try:
        # reasoning_agent = state.get("reasoning_model")
        sub_claims = state.get("sub_claims", [])
        retrieved_docs = state.get("retrieved_docs", {})
        
        reasoning_outputs = {}
        
        # Iteriamo su ogni sub-claim per generare il ragionamento
        for sc in sub_claims:
            # Placeholder: Qui chiamerai il modello Qwen passando il sub-claim e le evidenze
            docs = retrieved_docs.get(sc, "Nessuna evidenza trovata.")
            reasoning_outputs[sc] = (
                f"Sulla base dell'evidenza recuperata '{docs}', si evidenzia un allineamento "
                f"parziale/totale con i concetti espressi nel claim."
            )
            
        print(f"✅ Generata Chain-of-Thought per {len(reasoning_outputs)} sub-claims.")
        return {"reasoning_output": reasoning_outputs, "reasoning_checked": True}
    except Exception as e:
        print(f"❌ Errore durante il ragionamento: {e}")
        return {"reasoning_output": {}, "reasoning_checked": False}