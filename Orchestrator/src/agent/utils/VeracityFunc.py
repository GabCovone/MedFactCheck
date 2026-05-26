from typing import Dict, Any

async def veracity_output_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Controlla che la valutazione della veridicità abbia prodotto risultati validi."""
    print("--- CHECKING VERACITY OUTPUT ---")
    veracity_results = state.get("veracity_results")
    if veracity_results and isinstance(veracity_results, dict) and len(veracity_results) > 0:
        print(f"✅ Check superato: Verdetti calcolati per {len(veracity_results)} sub-claims.")
        return {"veracity_checked": True}
    else:
        print("❌ Check fallito: Nessun verdetto calcolato, impossibile procedere.")
        return {"veracity_checked": False}

async def run_veracity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determina il verdetto finale (Supported, Refuted, NEI) e lo score di confidenza
    basandosi sui sub-claims, le evidenze e la Chain-of-Thought.
    """
    print("--- DETERMINING VERACITY ---")
    try:
        veracity_agent = state.get("veracity_model")
        sub_claims = state.get("sub_claims", [])
        reasoning_output = state.get("reasoning_output", {})
        retrieved_docs = state.get("retrieved_docs", {})
        
        veracity_results = {}
        
        for sc in sub_claims:
            cot = reasoning_output.get(sc, "")
            docs = retrieved_docs.get(sc, [])

            final_label, score = veracity_agent.assess_veracity(
                sub_claim=sc,
                evidence_list=docs,
                reasoning_text=cot
            )

            # Costruiamo il dizionario esattamente come se lo aspetta StorageManager.save_verdict()
            veracity_results[sc] = {
                "claim": sc,
                "verdict": final_label, 
                "confidence_score": score,
                "chain_of_thought_log": cot,
                "supporting_evidence": docs
            }
            
        print(f"✅ Calcolato verdetto per {len(veracity_results)} sub-claims.")
        return {"veracity_results": veracity_results}
    except Exception as e:
        print(f"❌ Errore durante il calcolo della veridicità: {e}")
        return {"veracity_results": {}}