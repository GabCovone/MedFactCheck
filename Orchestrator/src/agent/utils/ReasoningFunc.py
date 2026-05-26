from typing import Dict, Any

async def reasoning_output_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Controlla che il ragionamento abbia prodotto risultati validi."""
    print("--- CHECKING REASONING OUTPUT ---")
    reasoning_output = state.get("reasoning_output")
    if reasoning_output and isinstance(reasoning_output, dict) and len(reasoning_output) > 0:
        print(f"✅ Check superato: Ragionamento generato per {len(reasoning_output)} sub-claims.")
        return {"reasoning_checked": True}
    else:
        print("❌ Check fallito: Generazione del ragionamento fallita o vuota.")
        return {"reasoning_checked": False}

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
        return {"reasoning_output": reasoning_outputs}
    except Exception as e:
        print(f"❌ Errore durante il ragionamento: {e}")
        return {"reasoning_output": {}}