from typing import Dict, Any

async def input_to_json(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepara e valida l'input dell'utente per la decomposizione.
    Crea un dizionario con i campi richiesti dal modello ('text_input', 'image_path')
    e imposta un flag di successo ('json_checked').
    """
    print("--- PREPARING AND VALIDATING INPUT ---")
    
    claim_input = state.get("claim_input")
    
    # Validazione dell'input: deve esserci almeno testo o immagine.
    if not claim_input or not isinstance(claim_input, dict) or (not claim_input.get("text") and not claim_input.get("image")):
        print("❌ Errore: 'claim_input' non è valido o mancano sia 'text' che 'image'.")
        return {"json_checked": False}

    # Prepara il dizionario per il modello.
    # I campi corrispondono ai parametri di QwenNF4Decomposer.decompose.
    # Usa .get() per fornire un testo vuoto se non è presente, gestendo l'input solo immagine.
    decomposition_input = {"text_input": claim_input.get("text", "")}
    
    # Aggiunge il percorso dell'immagine se presente
    if "image" in claim_input and claim_input["image"]:
        decomposition_input["image_path"] = claim_input["image"]
        if claim_input.get("text"):
            print("✅ Input preparato con testo e immagine.")
        else:
            print("✅ Input preparato con solo immagine.")
    else:
        print("✅ Input preparato con solo testo.")

    # Restituisce il dizionario per il prossimo nodo e il flag di successo
    return {"decomposition_input": decomposition_input, "json_checked": True}


async def decompose_subclaims_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Controlla che il passo di decomposizione abbia prodotto una lista di sub-claims.
    """
    print("--- CHECKING DECOMPOSITION OUTPUT ---")
    
    sub_claims = state.get("sub_claims")
    
    if sub_claims and isinstance(sub_claims, list) and len(sub_claims) > 0:
        print(f"✅ Check superato: {len(sub_claims)} sub-claims trovati.")
        return {"input_checked": True}
    else:
        print("❌ Check fallito: Nessun sub-claim valido trovato nello stato.")
        return {"input_checked": False}

async def run_decomposition(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue la decomposizione del claim utilizzando l'agente caricato
    e l'input preparato dal nodo `input_to_json`.
    """
    print("--- RUNNING CLAIM DECOMPOSITION ---")
    try:
        decomposer_agent = state["decomposition_model"]
        prepared_input = state["decomposition_input"]
        
        decomposition_result = decomposer_agent.decompose(**prepared_input)
        print("Decomposition output received.")

        sub_claims_list = [sc["claim"] for sc in decomposition_result.get("sub_claims", [])]
        routing_info_dict = {sc["claim"]: sc["routes"] for sc in decomposition_result.get("sub_claims", [])}
        decomposer_reasoning = decomposition_result.get("reasoning", "")
        return {
            "sub_claims": sub_claims_list, 
            "routing_info": routing_info_dict, 
            "decomposer_reasoning": decomposer_reasoning
        }
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione della decomposizione: {e}")
        return {"sub_claims": [], "routing_info": {}}