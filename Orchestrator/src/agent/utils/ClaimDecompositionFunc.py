from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import json
import requests
import gc
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage

# IMPORTAZIONE CORRETTA PER QWEN 2.5
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


#Decomposer Agent
class BaseClaimDecomposer(ABC):
    """
    Classe astratta per la scomposizione dei claim.
    """

    @abstractmethod
    def decompose(self, text_input: str) -> Optional[Dict[str, Any]]:
        """
        Riceve in input il testo e restituisce un JSON strutturato
        con i sub-claims e le strategie di routing.
        """
        pass

class QwenNF4Decomposer(BaseClaimDecomposer):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Caricamento di {model_id} in modalità Multimodale (NF4 + SDPA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # CARICAMENTO CON SDPA (Ottimizzazione nativa della memoria)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa"
        )
        print("Modello Multimodale caricato e pronto!")

    def _is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _scrape_text_from_url(self, url: str) -> str:
        print(f"Scraping del contenuto da: {url}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64 AppleWebKit/537.36)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()

            text = soup.get_text(separator=' ', strip=True)
            return text[:4000] + ("..." if len(text) > 4000 else "")
        except Exception as e:
            print(f"Errore nello scraping URL: {e}")
            return f"Errore: Impossibile leggere {url}"

    def decompose(self, text_input: str, image_path: str = None) -> dict:
        if self._is_url(text_input):
            text_to_process = self._scrape_text_from_url(text_input)
            print("Testo estratto dall'URL con successo.")
        else:
            text_to_process = text_input

        # 1. Costruzione dei messaggi
        messages = build_qwen_few_shot_prompt(text_to_process, image_path)

        # 2. Template
        text_with_template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Gestione Immagine (Sblocco server, fix PNG trasparenti e BytesIO)
        if image_path:
            print(f"Caricamento immagine da: {image_path}...")
            try:
                if self._is_url(image_path):
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64 AppleWebKit/537.36)'}
                    img_response = requests.get(image_path, headers=headers, timeout=15)
                    img_response.raise_for_status()
                    image = Image.open(BytesIO(img_response.content)).convert("RGB")
                else:
                    image = Image.open(image_path).convert("RGB")

                inputs = self.processor(
                    text=[text_with_template],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)
            except Exception as e:
                print(f"Errore caricamento immagine: {e}")
                inputs = self.processor(text=[text_with_template], return_tensors="pt").to(self.model.device)
        else:
            # IL BLOCCO CHE MANCAVA! Questo gestisce l'input solo testuale.
            inputs = self.processor(text=[text_with_template], return_tensors="pt").to(self.model.device)

        # 4. Generazione
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )

        # 5. Decodifica
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 6. Parsing
        risultato_json = self._parse_json_output(generated_text)

        # 7. Pulizia Memoria VRAM
        del inputs
        del generated_ids
        del generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()

        return risultato_json

    def _parse_json_output(self, raw_text: str) -> dict:
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.replace(".\n  \"sub_claims\"", ".\",\n  \"sub_claims\"")

        try:
            parsed_json = json.loads(clean_text.strip())
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"ERRORE DI PARSING JSON: {e}")
            print(f"Testo grezzo generato:\n{raw_text}")
            return {"sub_claims": []}


#PROMPT BUILDING
def build_qwen_few_shot_prompt(user_text: str, image_path: str = None) -> list:
    """
    Costruisce i messaggi in formato standard per Qwen2.5-VL.
    Restituisce una lista di dizionari pronta per essere processata da apply_chat_template().
    """

    # --- 1. SYSTEM PROMPT (Identità Strict e Regole per Classificazioni) ---
    system_prompt = """Sei uno script di estrazione dati MULTIMODALE e AUTOMATICO. Puoi analizzare sia testo che immagini. Il tuo unico scopo è estrarre un array di proposizioni dichiarative indipendenti (Soggetto + Verbo + Oggetto).
NON sei un assistente conversazionale. NON scusarti, NON dialogare e NON fare domande. Restituisci SEMPRE e SOLO il JSON valido. Se rifiuti di analizzare, lo script andrà in crash.

REGOLE SINTATTICHE (CRITICHE):
1. COMPLETEZZA: Estrai TUTTI i claim medici, chimici o scientifici rilevanti dal testo o dai contenuti dell'immagine.
2. ANEDDOTI E FAKE NEWS: Scarta rigorosamente le narrazioni personali ("Mio cugino ha preso..."). Tuttavia, se la frase nasconde una teoria medica (es. "X è la cura definitiva!"), ESTRAI QUELLA TEORIA in modo neutrale.
3. RISOLUZIONE DEL SOGGETTO: I pronomi o i soggetti sottintesi devono essere esplicitati.
4. REASONING BREVE: Il campo "reasoning" deve essere di MASSIMO 15 PAROLE per evitare errori di formattazione.

REGOLE DI CLASSIFICAZIONE (ROUTING) - LEGGERE ATTENTAMENTE:
- Assegna ["kb"] SOLO alle frasi che descrivono definizioni biologiche/chimiche, tassonomia, composizione o proprietà puramente statiche (es. "X è un ormone", "L'aspirina è un farmaco", "X contiene Y", "X ha una massa di Z").
- Assegna ["kb", "lit"] a TUTTE le altre frasi che descrivono cause, correlazioni, azioni, effetti clinici, cure o eventi (es. "X altera Y", "X riduce Z", "X è la causa di Y").

FORMATO DI OUTPUT:
{
  "reasoning": "Breve logica...",
  "sub_claims": [
    {
      "claim": "Frase completa e indipendente",
      "routes": ["..."]
    }
  ]
}"""

    # --- 2. LA STRUTTURA A MESSAGGI (Few-Shot Examples) ---
    messages = [
        # Inizializziamo il System Prompt
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},

        # ESEMPIO 1: Routing base
        {"role": "user", "content": [{"type": "text", "text": "TITOLO: La radice magica. PUNTO 1: Contiene molta vitamina C. PUNTO 2: Cura il cancro ai polmoni in una settimana."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Punto 1 composizione (kb). Punto 2 azione estrema (kb, lit). Soggetto risolto.\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"La radice magica contiene molta vitamina C.\",\n      \"routes\": [\"kb\"]\n    },\n    {\n      \"claim\": \"La radice magica cura il cancro ai polmoni in una settimana.\",\n      \"routes\": [\"kb\", \"lit\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 2: Eliminazione metodologie
        {"role": "user", "content": [{"type": "text", "text": "Uno studio dimostra che mangiare aglio altera i batteri intestinali. Ricordiamo che l'aglio è un bulbo."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Scarto la metodologia. Estraggo l'azione (kb, lit) e la definizione statica (kb).\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"Mangiare aglio altera i batteri intestinali.\",\n      \"routes\": [\"kb\", \"lit\"]\n    },\n    {\n      \"claim\": \"L'aglio è un bulbo.\",\n      \"routes\": [\"kb\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 3: Gestione severa di Aneddoti e Fake News mischiate (Questo risolve il Test 2)
        {"role": "user", "content": [{"type": "text", "text": "Mio nonno mangiava sempre la terra e il giorno dopo stava bene. I medici mentono, la terra è la cura definitiva! Ricordate che la terra contiene minerali."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Scarto aneddoto del nonno. Estraggo claim medico estremo (kb, lit) e composizione (kb).\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"La terra è la cura definitiva.\",\n      \"routes\": [\"kb\", \"lit\"]\n    },\n    {\n      \"claim\": \"La terra contiene minerali.\",\n      \"routes\": [\"kb\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 4: Routing per classificazioni farmaceutiche (Questo risolve il Test 6)
        {"role": "user", "content": [{"type": "text", "text": "L'ibuprofene è un antinfiammatorio non steroideo. Riduce drasticamente il dolore articolare."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Estraggo classificazione farmacologica (kb) e azione clinica (kb, lit).\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"L'ibuprofene è un antinfiammatorio non steroideo.\",\n      \"routes\": [\"kb\"]\n    },\n    {\n      \"claim\": \"L'ibuprofene riduce drasticamente il dolore articolare.\",\n      \"routes\": [\"kb\", \"lit\"]\n    }\n  ]\n}"}]}
    ]

    # --- 3. GESTIONE DELL'INPUT REALE DELL'UTENTE (Con Multimodalità) ---
    user_content = []

    # Se c'è un'immagine, la aggiungiamo PRIMA del testo
    if image_path:
        user_content.append({"type": "image", "image": image_path})

    # Aggiungiamo il testo reale
    user_content.append({"type": "text", "text": user_text})

    # Aggiungiamo il messaggio dell'utente alla lista
    messages.append({"role": "user", "content": user_content})

    return messages


async def init_decomposition(state: Dict[str, Any]) -> Dict[str, Any]:
    decomposer_agent = QwenNF4Decomposer()
    print(f"Decomposition init: ", decomposer_agent)
    return {"decomposition_model": decomposer_agent}

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
