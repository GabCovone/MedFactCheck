from typing import Dict, Any
from langchain_core.messages import AIMessage

class ReasonerAgent:
    """Agente Reasoner per l'architettura Multi-Agente in approccio OOP."""
    def __init__(self, reasoning_model: Any):
        self.qwen_instance = reasoning_model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[AGENTE REASONER] Genero il ragionamento logico (CoT) basato sulle evidenze.")
        sub_claims = state.get("sub_claims", [])
        retrieved_docs = state.get("retrieved_docs", {})
        
        reasoning_outputs = {}
        for sc in sub_claims:
            docs = retrieved_docs.get(sc, [])
            
            # --- EDGE CASE CHECK: Se non ci sono documenti, evitiamo di chiamare il LLM ---
            if not docs:
                print(f" -> ⚠️ Nessuna evidenza per '{sc[:30]}...'. Salto l'inferenza.")
                reasoning_outputs[sc] = "Non Enough Information (NEI): Nessuna evidenza recuperata dalle fonti disponibili per supportare o confutare questo claim."
                continue
            # -------------------------------------------------------------------------------
            
            print(f" -> Elaboro CoT per '{sc[:30]}...' con {len(docs)} documenti.")
            
            # Inferenza gerarchica iterativa: divisione in batch di max 4 documenti
            batch_size = 4
            current_docs = docs
            level = 1
            
            # Riduciamo i documenti finché non sono <= batch_size
            while len(current_docs) > batch_size:
                print(f"    - Map-Reduce Livello {level}: Compatto {len(current_docs)} documenti in batch da {batch_size}...")
                intermediate_docs = []
                for i in range(0, len(current_docs), batch_size):
                    batch = current_docs[i:i + batch_size]
                    batch_cot = self.qwen_instance.reason(sub_claim=sc, evidence_list=batch)
                    intermediate_docs.append({"text": batch_cot, "source": f"Intermediate Analysis L{level}-{i//batch_size + 1}"})
                current_docs = intermediate_docs
                level += 1
                
            # Ragionamento finale (o unica inferenza diretta se i documenti erano già <= 4)
            print(f"    - Genero CoT finale da {len(current_docs)} documenti distillati.")
            cot = self.qwen_instance.reason(sub_claim=sc, evidence_list=current_docs)
            
            reasoning_outputs[sc] = cot
            
        log_details = "\n".join([f"  - '{sc[:60]}...' -> Generato CoT ({len(cot.split())} parole)." for sc, cot in reasoning_outputs.items()])
        log_message = f"Ho completato il ragionamento logico (Chain-of-Thought) basato sulle evidenze per {len(reasoning_outputs)} sub-claims:\n{log_details}\nSi può procedere con il modulo di Veracity."
        return {
            "reasoning_output": reasoning_outputs,
            "messages": [AIMessage(content=log_message, name="Reasoner")]
        }