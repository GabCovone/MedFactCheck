from abc import ABC, abstractmethod
import torch
import json
import requests
import gc
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from qwen_vl_utils import process_vision_info

# IMPORTAZIONE CORRETTA PER QWEN 2.5
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Decomposer Agent
class Base_Qwen(ABC):
    @abstractmethod
    def decompose(self, text_input: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def reason(self, text_input: str) -> Optional[Dict[str, Any]]:
        pass

class QwenNF4(Base_Qwen):
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

    # AGGIUNTO 'self' QUI!
    def build_qwen_few_shot_prompt(self, user_text: str, image_path: str = None) -> list:
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

        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": user_content})

        return messages

    def decompose(self, text_input: str, image_path: str = None) -> dict:
        if self._is_url(text_input):
            text_to_process = self._scrape_text_from_url(text_input)
            print("Testo estratto dall'URL con successo.")
        else:
            text_to_process = text_input

        # 1. Costruzione dei messaggi
        messages = self.build_qwen_few_shot_prompt(text_to_process, image_path)

        # 2. Template e Vision Info (Codice uniformato e pulito!)
        text_with_template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_with_template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 3. Generazione
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )

        # 4. Decodifica
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 5. Parsing
        risultato_json = self._parse_json_output(generated_text)

        # 6. Pulizia Memoria VRAM
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
            return {"sub_claims": []}
        
    def reason(self, sub_claim, evidence_list, image_path=None):
        evidence_text = "\n".join([f"- {e['text']} (Fonte: {e['source']})" for e in evidence_list])

        prompt_text = f"""Sei un esperto di fact-checking biomedico. Analizza il seguente claim basandoti sulle evidenze fornite e sull'immagine allegata.
        Genera un ragionamento passo-passo (Chain-of-Thought)...
        CLAIM: {sub_claim}
        EVIDENZE SCIENTIFICHE:
        {evidence_text}
        RAGIONAMENTO LOGICO:"""

        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Sei un analista scientifico rigoroso e imparziale."}]},
            {"role": "user", "content": user_content}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Pulizia VRAM anche qui!
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        return response.strip()
    


async def init_qwen_instance(state: Dict[str, Any]) -> Dict[str, Any]:
    qwen_instance = QwenNF4()
    return {"decomposition_model": qwen_instance, "reasoning_model": qwen_instance}