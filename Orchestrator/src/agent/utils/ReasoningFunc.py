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
        reasoning_agent = state.get("reasoning_model")
        sub_claims = state.get("sub_claims", [])
        retrieved_docs = state.get("retrieved_docs", {})
        
        reasoning_outputs = {}
        
        # Iteriamo su ogni sub-claim per generare il ragionamento
        for sc in sub_claims:
            docs = retrieved_docs.get(sc, [])
            
            # Siccome il retrieval al momento è mockato e restituisce stringhe in graph.py,
            # lo normalizziamo in una lista di dizionari come si aspetta la classe Qwen
            if isinstance(docs, str):
                docs = [{"text": docs, "source": "Placeholder"}]
                
            # Inferenza gerarchica iterativa: divisione in batch di max 4 documenti
            batch_size = 4
            current_docs = docs
            level = 1
            
            # Riduciamo i documenti finché non sono <= batch_size
            while len(current_docs) > batch_size:
                intermediate_docs = []
                for i in range(0, len(current_docs), batch_size):
                    batch = current_docs[i:i + batch_size]
                    batch_cot = reasoning_agent.reason(sub_claim=sc, evidence_list=batch)
                    intermediate_docs.append({"text": batch_cot, "source": f"Intermediate Analysis L{level}-{i//batch_size + 1}"})
                current_docs = intermediate_docs
                level += 1
                
            # Ragionamento finale (o unica inferenza diretta se i documenti erano già <= 4)
            cot = reasoning_agent.reason(sub_claim=sc, evidence_list=current_docs)
            
            reasoning_outputs[sc] = cot
            
        print(f"✅ Generata Chain-of-Thought per {len(reasoning_outputs)} sub-claims.")
        return {"reasoning_output": reasoning_outputs, "reasoning_checked": True}
    except Exception as e:
        print(f"❌ Errore durante il ragionamento: {e}")
        return {"reasoning_output": {}, "reasoning_checked": False}