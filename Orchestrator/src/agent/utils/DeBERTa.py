import torch
import gc
from typing import Dict, List, Optional, Tuple
from transformers import pipeline

class DebertaVeracityNode:
    """
    Agente specializzato nella classificazione NLI (Natural Language Inference).
    Valuta se le evidenze supportano (entailment) o smentiscono (contradiction) un claim.
    """
    def __init__(self, model_id: str = "cross-encoder/nli-deberta-v3-base"):
        print(f"Caricamento del classificatore di veridicità: {model_id}...")
        
        # Determiniamo il dispositivo ottimale (GPU se disponibile, altrimenti CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Inizializziamo la pipeline in float16 per risparmiare memoria sulla GPU
        self.classifier = pipeline(
            "text-classification",
            model=model_id,
            device=self.device,
            torch_dtype=torch.float16 if self.device == 0 else torch.float32,
            truncation=True,
            max_length=512
        )
        print("Classificatore DeBERTa caricato con successo!")

    def assess_veracity(self, sub_claim: str, evidence_list: List[Dict[str, str]], reasoning_text: str) -> Tuple[str, float]:
        """
        Valuta la veridicità di un claim basandosi sulle evidenze e sul ragionamento generato.
        
        Args:
            sub_claim (str): L'affermazione da verificare.
            evidence_list (list): Lista di dizionari contenenti le evidenze trovate.
            reasoning_text (str): Il ragionamento logico prodotto dall'agente di reasoning.
            
        Returns:
            Tuple[str, float]: Una tupla contenente il verdetto finale ('Supported', 'Refuted', 'Not Enough Information')
                               e lo score di confidenza (0.0 - 1.0).
        """
        # 1. Costruiamo il contesto aggregando tutte le evidenze
        # Assumiamo che i dizionari in evidence_list abbiano una chiave 'testo' o 'text'
        evidence_all = " ".join([
            e.get("testo", e.get("text", "")) for e in evidence_list
        ])
        
        # 2. Assembliamo la stringa di contesto (Premessa) e l'ipotesi
        context = f"Scientific Evidence: {evidence_all} Analysis: {reasoning_text}"
        hypothesis = f"Based on this, it is true that {sub_claim}"

        # 3. Eseguiamo la classificazione
        result = self.classifier([{"text": context, "text_pair": hypothesis}])[0]

        raw_label = result["label"].lower()
        score = result["score"]

        # 4. Pulizia manuale della memoria
        del context
        del hypothesis
        del result
        
        if self.device == 0:
            torch.cuda.empty_cache()
        gc.collect()

        # 5. Mappiamo le etichette del modello (NLI standard) ai nostri verdetti
        # I modelli NLI solitamente restituiscono: entailment (0), neutral (1), contradiction (2)
        # o stringhe come "LABEL_0", "LABEL_1", "LABEL_2" a seconda di come sono stati salvati.
        if raw_label in ["entailment", "label_0"]:
            final_label = "Supported"
        elif raw_label in ["contradiction", "label_2"]:
            final_label = "Refuted"
        else:
            final_label = "Not Enough Information"

        return final_label, round(score, 4)

    def format_final_report(self, claim: str, verdict: str, confidence: float, reasoning: str, evidence: List[Dict]) -> Dict:
        """
        Struttura il risultato finale nel formato JSON standard per l'output.
        """
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence_score": confidence,
            "chain_of_thought_log": reasoning,
            "supporting_evidence": evidence
        }


async def init_deberta_istance():
    deberta_istance = DebertaVeracityNode()
    return deberta_istance