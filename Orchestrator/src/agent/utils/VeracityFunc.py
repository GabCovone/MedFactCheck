from typing import Dict, Any

async def veracity_input_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Controlla che i dati necessari per la veracity (sub_claims e reasoning_output) siano presenti."""
    print("--- CHECKING VERACITY INPUT ---")
    sub_claims = state.get("sub_claims")
    reasoning_output = state.get("reasoning_output")
    
    if sub_claims and isinstance(sub_claims, list) and isinstance(reasoning_output, dict):
        print("✅ Check superato: Input valido per la classificazione di veridicità.")
        return {"veracity_input_checked": True}
    else:
        print("❌ Check fallito: Input per la veracity mancante o non valido.")
        return {"veracity_input_checked": False}

async def run_veracity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determina il verdetto finale (Supported, Refuted, NEI) e lo score di confidenza
    basandosi sui sub-claims, le evidenze e la Chain-of-Thought.
    """
    print("--- DETERMINING VERACITY ---")
    try:
        # veracity_agent = state.get("veracity_model")
        sub_claims = state.get("sub_claims", [])
        reasoning_output = state.get("reasoning_output", {})
        
        veracity_results = {}
        
        for sc in sub_claims:
            cot = reasoning_output.get(sc, "")
            # Placeholder: Qui chiamerai il modello DeBERTa passandogli claim + CoT
            # Costruiamo il dizionario esattamente come se lo aspetta StorageManager.save_verdict()
            veracity_results[sc] = {
                "claim": sc,
                "verdict": "Supported", 
                "confidence_score": 0.854,
                "chain_of_thought_log": cot,
                "supporting_evidence": []
            }
            
        print(f"✅ Calcolato verdetto per {len(veracity_results)} sub-claims.")
        return {"veracity_results": veracity_results, "veracity_checked": True}
    except Exception as e:
        print(f"❌ Errore durante il calcolo della veridicità: {e}")
        return {"veracity_results": {}, "veracity_checked": False}