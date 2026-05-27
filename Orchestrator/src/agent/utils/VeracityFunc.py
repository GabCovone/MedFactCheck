from typing import Dict, Any
from langchain_core.messages import AIMessage

class VeracityAgent:
    """
    Agente Veracity per l'architettura Multi-Agente.
    Usa Short-Circuit evaluation per ottimizzare i calcoli.
    """
    def __init__(self, veracity_model: Any):
        self.deberta_instance = veracity_model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[AGENTE VERACITY] Emetto i verdetti finali e valuto la confidenza (NLI).")
        sub_claims = state.get("sub_claims", [])
        reasoning_output = state.get("reasoning_output", {})
        retrieved_docs = state.get("retrieved_docs", {})
        
        veracity_results = {}
        for sc in sub_claims:
            cot = reasoning_output.get(sc, "")
            docs = retrieved_docs.get(sc, [])
            
            # --- SHORT-CIRCUIT: Nessun documento = Niente inferenza pesante ---
            if not docs:
                print(f" -> ⚠️ Nessuna evidenza per '{sc[:30]}...'. Assegno verdetto NEI.")
                veracity_results[sc] = {
                    "claim": sc,
                    "verdict": "Not Enough Information", 
                    "confidence_score": 1.0, # 100% certi che manchino i dati
                    "chain_of_thought_log": "Nessuna evidenza recuperata per valutare logicamente questo claim.",
                    "supporting_evidence": []
                }
                continue
            # ------------------------------------------------------------------
            
            print(f" -> Valuto NLI per: '{sc[:30]}...'")
            final_label, score = self.deberta_instance.assess_veracity(
                sub_claim=sc,
                reasoning_text=cot
            )
            
            veracity_output = {
                "claim": sc,
                "verdict": final_label, 
                "confidence_score": score,
                "chain_of_thought_log": cot,
                "supporting_evidence": docs # Li manteniamo nel JSON finale per la Dashboard!
            }
            veracity_results[sc] = veracity_output
            
        return {
            "veracity_results": veracity_results,
            "messages": [AIMessage(content=f"Calcolati {len(veracity_results)} verdetti finali con successo.", name="Veracity")]
        }