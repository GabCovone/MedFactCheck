import torch
import gc
from typing import Dict, List, Tuple
from transformers import pipeline

class DebertaVeracityNode:
    """
    Agente specializzato nella classificazione NLI (Natural Language Inference).
    Valuta se il ragionamento logico (CoT) supporta o smentisce un claim.
    """
    def __init__(self, model_id: str = "cross-encoder/nli-deberta-v3-base"):
        print(f"Caricamento del classificatore NLI: {model_id}...")
        
        # Dispositivo ottimale
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Inizializzazione pipeline FP16
        self.classifier = pipeline(
            "text-classification",
            model=model_id,
            device=self.device,
            torch_dtype=torch.float16 if self.device == 0 else torch.float32,
            truncation=True,
            max_length=512
        )
        print("Classificatore DeBERTa caricato con successo!")

    def assess_veracity(self, sub_claim: str, reasoning_text: str) -> Tuple[str, float]:
        """
        Valuta la veridicità usando SOLO il CoT come premessa e il claim come ipotesi.
        """
        # Protezione anti-crash: se manca il reasoning o indica dati insufficienti
        if not reasoning_text or "Not Enough Information" in reasoning_text or "Non Enough Information" in reasoning_text:
            return "Not Enough Information", 1.0

        # Assembliamo Premessa (CoT) e Ipotesi (Claim) in modo super compatto
        context = f"Logical Analysis: {reasoning_text}"
        hypothesis = f"Therefore, it is true that: {sub_claim}"

        # Classificazione NLI
        result = self.classifier([{"text": context, "text_pair": hypothesis}])[0]

        raw_label = result["label"].lower()
        score = result["score"]

        # Pulizia rigorosa VRAM
        del context, hypothesis, result
        if self.device == 0:
            torch.cuda.empty_cache()
        gc.collect()

        # Mapping NLI Standard per i verdetti
        if raw_label in ["entailment", "label_0"]:
            final_label = "Supported"
        elif raw_label in ["contradiction", "label_2"]:
            final_label = "Refuted"
        else:
            final_label = "Not Enough Information"

        return final_label, round(score, 4)