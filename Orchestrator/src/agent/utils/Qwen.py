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
    def decide_tool(self, task_context: str, available_tools: list) -> list:
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
        system_prompt = """You are an AUTOMATIC and MULTIMODAL data extraction script. You can analyze both text and images. Your only purpose is to extract an array of independent declarative propositions (Subject + Verb + Object).
        You are NOT a conversational assistant. DO NOT apologize, DO NOT converse, and DO NOT ask questions. ALWAYS and ONLY return a valid JSON. If you refuse to analyze, the script will crash.

        SYNTACTIC RULES (CRITICAL):
        1. COMPLETENESS: Extract ALL relevant medical, chemical, or scientific claims from the text or image contents.
        1. EXTRACTION LIMIT: Extract a MAXIMUM of 5 most important medical, chemical, or scientific sub-claims from the text or image contents. Do not extract every single detail.
        2. ANECDOTES AND FAKE NEWS: Strictly discard personal narratives ("My cousin took..."). However, if the sentence hides a medical theory (e.g., "X is the definitive cure!"), EXTRACT THAT THEORY in a neutral way.
        3. SUBJECT RESOLUTION: Pronouns or implied subjects must be made explicit.
        4. SHORT REASONING: The "reasoning" field must be MAXIMUM 15 WORDS to avoid formatting errors.

        CLASSIFICATION RULES (ROUTING) - READ CAREFULLY:
        - Assign ["kb"] ONLY to sentences describing biological/chemical definitions, taxonomy, composition, or purely static properties (e.g., "X is a hormone", "Aspirin is a drug", "X contains Y", "X has a mass of Z").
        - Assign ["kb", "lit"] to ALL other sentences describing causes, correlations, actions, clinical effects, cures, or events (e.g., "X alters Y", "X reduces Z", "X is the cause of Y").

        OUTPUT FORMAT:
        {
        "reasoning": "Short logic...",
        "sub_claims": [
            {
            "claim": "Complete and independent sentence",
            "routes": ["..."]
            }
        ]
        }"""
        
        messages = [
        # Inizializziamo il System Prompt
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},

        # ESEMPIO 1: Routing base
        {"role": "user", "content": [{"type": "text", "text": "TITLE: The magic root. POINT 1: It contains a lot of vitamin C. POINT 2: It cures lung cancer in a week."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Point 1 composition (kb). Point 2 extreme action (kb, lit). Subject resolved.\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"The magic root contains a lot of vitamin C.\",\n      \"routes\": [\"kb\"]\n    },\n    {\n      \"claim\": \"The magic root cures lung cancer in a week.\",\n      \"routes\": [\"kb\", \"lit\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 2: Eliminazione metodologie
        {"role": "user", "content": [{"type": "text", "text": "A study shows that eating garlic alters intestinal bacteria. Recall that garlic is a bulb."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Discard methodology. Extract action (kb, lit) and static definition (kb).\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"Eating garlic alters intestinal bacteria.\",\n      \"routes\": [\"kb\", \"lit\"]\n    },\n    {\n      \"claim\": \"Garlic is a bulb.\",\n      \"routes\": [\"kb\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 3: Gestione severa di Aneddoti e Fake News mischiate (Questo risolve il Test 2)
        {"role": "user", "content": [{"type": "text", "text": "My grandfather always ate dirt and the next day he was fine. Doctors lie, dirt is the definitive cure! Remember that dirt contains minerals."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Discard grandfather anecdote. Extract extreme medical claim (kb, lit) and composition (kb).\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"Dirt is the definitive cure.\",\n      \"routes\": [\"kb\", \"lit\"]\n    },\n    {\n      \"claim\": \"Dirt contains minerals.\",\n      \"routes\": [\"kb\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 4: Routing per classificazioni farmaceutiche (Questo risolve il Test 6)
        {"role": "user", "content": [{"type": "text", "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug. It drastically reduces joint pain."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"Extract pharmacological classification (kb) and clinical action (kb, lit).\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"Ibuprofen is a nonsteroidal anti-inflammatory drug.\",\n      \"routes\": [\"kb\"]\n    },\n    {\n      \"claim\": \"Ibuprofen drastically reduces joint pain.\",\n      \"routes\": [\"kb\", \"lit\"]\n    }\n  ]\n}"}]},

        # ESEMPIO 5: Contraddizione Multimodale (Insegna al modello a smentire l'utente guardando l'immagine)
        {"role": "user", "content": [{"type": "text", "text": "USER TEXT CLAIM: 'This is a fresh and perfectly healthy apple.'\nCRITICAL SYSTEM INSTRUCTION: You MUST also analyze the attached image. If the image visually contradicts the USER TEXT CLAIM, you must extract the true visual evidence as a sub-claim."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{\n  \"reasoning\": \"The text claims the apple is healthy, but visual analysis shows it is rotten. Extract both to allow fact-checking of the contradiction.\",\n  \"sub_claims\": [\n    {\n      \"claim\": \"This is a fresh and perfectly healthy apple.\",\n      \"routes\": [\"kb\"]\n    },\n    {\n      \"claim\": \"The attached visual evidence shows a rotten, decaying apple with irregular brown spots.\",\n      \"routes\": [\"kb\", \"lit\"]\n    }\n  ]\n}"}]}
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
                max_new_tokens=4096,
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

        prompt_text = f"""You are an expert biomedical analyst tasked with validating scientific claims.
        Evaluate the following claim by rigorously comparing it with the provided evidence.
        
        CLAIM TO VERIFY: "{sub_claim}"
        
        SCIENTIFIC EVIDENCE FOUND:
        {evidence_text}
        
        REASONING INSTRUCTIONS (Chain-of-Thought):
        Write a detailed and discursive analysis (about 100-150 words). In your text you must:
        1. Synthesize what the provided scientific evidence clearly states.
        2. Explicitly relate the concepts of the evidence to the claim.
        3. Conclude unequivocally whether the evidence supports, refutes, or does not offer sufficient details to confirm the claim.
        Do not use bullet points, write a logical and cohesive paragraph.
        
        DETAILED LOGICAL ANALYSIS:"""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a rigorous, verbose, and impartial scientific analyst. Your goal is to explain the logical reasoning in detail."}]},
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
        system_prompt = """You are the Supervisor of a medical Multi-Agent system (MedFactCheck).
        Your task is to read the message history and decide which agent should be executed in the next step.
        The available agents are: "Decomposer", "Retriever", "Reasoner", "Veracity", "FINISH".
        You must reply EXCLUSIVELY with the exact name of the agent or "FINISH". No other words or punctuation."""
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": "No operations so far. The claim just arrived."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Decomposer"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Decomposer]: I successfully generated 2 sub-claims. Now you can search for evidence."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Retriever"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Retriever]: Evidence found for 2 sub-claims."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Reasoner"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Reasoner]: I completed the logical reasoning (CoT) for the sub-claims. We can proceed with veracity."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Veracity"}]},
            {"role": "user", "content": [{"type": "text", "text": "[Veracity]: I calculated and saved the final verdicts. The claim has been completely processed."}]},
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
        system_prompt = f"""You are an autonomous decision maker in a medical tool-calling system. Your task is to select the most appropriate tools to solve the current task based on the provided inputs.
        The available tools are: [{tools_str}].
        Reply EXCLUSIVELY with a valid JSON array containing the NAMES OF THE TOOLS chosen, without any additional explanation."""
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            
            # Esempi per Agente Ingestion
            {"role": "user", "content": [{"type": "text", "text": "Task: The user provided this raw text input: 'https://en.wikipedia.org/wiki/Paracetamol...'. Select 'scrape_text_from_url' if the text is a web link. Select 'validate_image' if an image path is provided or if the text itself is an image path. Select 'process_plain_text' if it's a normal medical claim. You can select multiple tools if needed."}]},
            {"role": "assistant", "content": [{"type": "text", "text": '["scrape_text_from_url"]'}]},
            {"role": "user", "content": [{"type": "text", "text": "Task: The user provided this raw text input: '/home/user/images/chest_xray.png...'. Select 'scrape_text_from_url' if the text is a web link. Select 'validate_image' if an image path is provided or if the text itself is an image path. Select 'process_plain_text' if it's a normal medical claim. You can select multiple tools if needed."}]},
            {"role": "assistant", "content": [{"type": "text", "text": '["validate_image"]'}]},
            {"role": "user", "content": [{"type": "text", "text": "Task: The user provided this raw text input: 'Aspirin cures headaches...'. Select 'scrape_text_from_url' if the text is a web link. Select 'validate_image' if an image path is provided or if the text itself is an image path. Select 'process_plain_text' if it's a normal medical claim. You can select multiple tools if needed."}]},
            {"role": "assistant", "content": [{"type": "text", "text": '["process_plain_text"]'}]},
            {"role": "user", "content": [{"type": "text", "text": "Task: The user provided this raw text input: 'Aspirin cures headaches...' and this image path: '/tmp/img.png'. Select 'scrape_text_from_url' if the text is a web link. Select 'validate_image' if an image path is provided or if the text itself is an image path. Select 'process_plain_text' if it's a normal medical claim. You can select multiple tools if needed."}]},
            {"role": "assistant", "content": [{"type": "text", "text": '["process_plain_text", "validate_image"]'}]},

            {"role": "user", "content": [{"type": "text", "text": f"Task: {task_context}"}]}
        ]
        return messages

    def decide_tool(self, task_context: str, available_tools: list) -> list:
        messages = self.build_tool_decision_prompt(task_context, available_tools)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=30, temperature=0.1, do_sample=False
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            clean_resp = response.strip()
            if clean_resp.startswith("```json"): clean_resp = clean_resp[7:]
            if clean_resp.startswith("```"): clean_resp = clean_resp[3:]
            if clean_resp.endswith("```"): clean_resp = clean_resp[:-3]
            tools = json.loads(clean_resp.strip())
            if isinstance(tools, list): return tools
            return []
        except json.JSONDecodeError:
            print(f"⚠️ Errore parsing tool JSON: {response}")
            return []