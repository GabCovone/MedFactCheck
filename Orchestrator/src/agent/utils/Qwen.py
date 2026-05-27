from abc import ABC, abstractmethod
from io import BytesIO
import requests
import torch
import json
import gc
from PIL import Image
from typing import Dict, Any, Optional

# IMPORTAZIONE CORRETTA PER QWEN 2.5
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Decomposer Agent
class Base_Qwen(ABC):
    @abstractmethod
    def decompose(self, text_input: str, image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def reason(self, sub_claim: str, evidence_list: list) -> str:
        pass

    @abstractmethod
    def supervise(self, messages_history: str) -> str:
        pass

    @abstractmethod
    def decide_tool(self, task_context: str, available_tools: list) -> str:
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
        # 1. Costruzione dei messaggi
        messages = self.build_qwen_few_shot_prompt(text_input, image_path)

        # 2. Template
        text_with_template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Gestione Immagine (Sblocco server, fix PNG trasparenti, BytesIO per URL)
        if image_path:
            print(f"Caricamento immagine da: {image_path}...")
            try:
                # Controlla se è un URL o un file locale
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    headers = {'User-Agent': 'Mozilla/5.0'}
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
                # Fallback in caso di errore di download o apertura
                inputs = self.processor(text=[text_with_template], padding=True, return_tensors="pt").to(self.model.device)
        else:
            # Gestione input solo testuale
            inputs = self.processor(text=[text_with_template], padding=True, return_tensors="pt").to(self.model.device)

        # 4. Generazione
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
            print(f"Testo grezzo generato:\n{raw_text}")
            return {"sub_claims": []}
        
    def reason(self, sub_claim: str, evidence_list: list) -> str:
        evidence_text = "\n".join([f"- {e['text']} (Fonte: {e['source']})" for e in evidence_list])

        prompt_text = f"""Sei un esperto analista biomedico incaricato di validare affermazioni scientifiche.
        Valuta il seguente claim confrontandolo rigorosamente con le evidenze fornite.
        
        CLAIM DA VERIFICARE: "{sub_claim}"
        
        EVIDENZE SCIENTIFICHE TROVATE:
        {evidence_text}
        
        ISTRUZIONI PER IL RAGIONAMENTO (Chain-of-Thought):
        Scrivi un'analisi dettagliata e discorsiva (circa 100-150 parole). Nel tuo testo devi:
        1. Sintetizzare cosa affermano chiaramente le evidenze scientifiche fornite.
        2. Mettere in relazione esplicita i concetti delle evidenze con il claim.
        3. Concludere in modo inequivocabile se le evidenze supportano, smentiscono o non offrono dettagli sufficienti per confermare il claim.
        Non usare elenchi puntati, scrivi un paragrafo logico e coeso.
        
        ANALISI LOGICA DETTAGLIATA:"""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Sei un analista scientifico rigoroso, verboso e imparziale. Il tuo scopo è spiegare il ragionamento logico in dettaglio."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]

        text_with_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text_with_template], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3, # Leggermente alzata per permettere un linguaggio più discorsivo e articolato
                do_sample=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Pulizia VRAM
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        return response.strip()

    def build_supervise_prompt(self, messages_history: str) -> list:
        system_prompt = """Sei il Supervisore di un sistema Multi-Agente medico (MedFactCheck).
        Il tuo compito è leggere la cronologia dei messaggi e decidere quale agente deve essere eseguito nel prossimo step.
        Gli agenti disponibili sono: "Decomposer", "Retriever", "Reasoner", "Veracity", "FINISH".
        Devi rispondere ESCLUSIVAMENTE con il nome esatto dell'agente o "FINISH". Nessun'altra parola o punteggiatura."""
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": "Nessuna operazione finora. Il claim è appena arrivato."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Decomposer"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Decomposer]: Ho generato 2 sub-claims con successo. Ora puoi cercare le evidenze."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Retriever"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Retriever]: Evidenze trovate per 2 sub-claims."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Reasoner"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Reasoner]: Ho completato il ragionamento logico (CoT) per i sub-claims. Si può procedere con la veridicità."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Veracity"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Veracity]: Ho calcolato e salvato i verdetti finali. Il claim è stato completamente processato."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "FINISH"}]},
            {"role": "user", "content": [{"type": "text", "text": messages_history}]}
        ]
        return messages

    def supervise(self, messages_history: str) -> str:
        messages = self.build_supervise_prompt(messages_history)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=10, temperature=0.1, do_sample=False
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        
        return response.strip()

    def build_tool_decision_prompt(self, task_context: str, available_tools: list) -> list:
        tools_str = ", ".join(available_tools)
        system_prompt = f"""Sei un decisore autonomo in un sistema di tool-calling medico. Il tuo compito è selezionare il tool più appropriato per risolvere il task corrente.
        I tool disponibili sono: [{tools_str}].
        Rispondi ESCLUSIVAMENTE con il NOME DEL TOOL scelto, senza alcuna spiegazione aggiuntiva. Se nessun tool è adatto, rispondi "nessuno"."""
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            
            # Esempi per Agente Ingestion
            {"role": "user", "content": [{"type": "text", "text": "Task: L'utente ha fornito questo input grezzo: 'https://it.wikipedia.org/wiki/Paracetamolo...'. Devi capire se è un link a un sito web, un percorso a un file immagine o un testo normale."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "scrape_text_from_url"}]},
            {"role": "user", "content": [{"type": "text", "text": "Task: L'utente ha fornito questo input grezzo: '/home/user/images/rx_torace.png...'. Devi capire se è un link a un sito web, un percorso a un file immagine o un testo normale."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "validate_image"}]},
            {"role": "user", "content": [{"type": "text", "text": "Task: L'utente ha fornito questo input grezzo: 'L'aspirina cura il mal di testa...'. Devi capire se è un link a un sito web, un percorso a un file immagine o un testo normale."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "nessuno"}]},

            {"role": "user", "content": [{"type": "text", "text": f"Task: {task_context}"}]}
        ]
        return messages

    def decide_tool(self, task_context: str, available_tools: list) -> str:
        messages = self.build_tool_decision_prompt(task_context, available_tools)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=15, temperature=0.1, do_sample=False
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        
        return response.strip()