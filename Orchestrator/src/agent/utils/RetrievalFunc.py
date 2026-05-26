from typing import Any, Dict


async def retrieve_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves evidence for sub-claims based on the routing info."""
    retriever = state.get('retrieval_models')
    sub_claims = state.get('sub_claims', [])
    routing_info = state.get('routing_info', {})

    print("--- RETRIEVING EVIDENCE ---")
    retrieved_docs = {}

    for claim in sub_claims:
        routes = routing_info.get(claim)
        print(f" -> Recupero evidenze per: '{claim}' [Rotte: {routes}]")
        evidence = retriever.retrieve(claim, routes)
        retrieved_docs[claim] = evidence

    print(f"Recuperate evidenze per {len(retrieved_docs)} sub-claims.")
    return {"retrieved_docs": retrieved_docs}

async def retrieval_output_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Controlla che i documenti recuperati (evidenze) per i sub_claims siano validi."""
    print("--- CHECKING RETRIEVAL OUTPUT ---")
    sub_claims = state.get("sub_claims")
    retrieved_docs = state.get("retrieved_docs")
    
    if sub_claims and isinstance(sub_claims, list) and isinstance(retrieved_docs, dict):
        print(f"✅ Check superato: Output del retrieval valido per {len(sub_claims)} sub-claims.")
        return {"retrieval_output_checked": True}
    else:
        print("❌ Check fallito: Output del retrieval mancante o non valido.")
        return {"retrieval_output_checked": False}