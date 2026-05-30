from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
from PIL import Image

@tool
def is_url(text: str) -> bool:
    """Verifica se una stringa è un URL valido."""
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

@tool
def validate_image(image_path: str) -> bool:
    """Verifica se il percorso (locale o URL) corrisponde a un'immagine valida per l'analisi."""
    if not image_path or not isinstance(image_path, str):
        return False

    # CASO 1: È un URL web?
    if image_path.startswith("http://") or image_path.startswith("https://"):
        try:
            # Usiamo .head() per scaricare SOLO le info, non l'immagine intera! Velocissimo.
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.head(image_path, headers=headers, timeout=5)
            # Controlla se il server dice che è un'immagine
            content_type = response.headers.get('Content-Type', '')
            return content_type.startswith('image/')
        except requests.RequestException:
            return False

    # CASO 2: È un file locale
    if not os.path.isfile(image_path):
        return False
        
    try:
        # verify() controlla l'header del file locale senza caricarlo in RAM
        with Image.open(image_path) as img:
            img.verify() 
        return True
    except Exception:
        return False

@tool
def scrape_text_from_url(url: str) -> str:
    """Estrae il testo da una pagina web dato un URL."""
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


class IngestionAndDecomposerAgent:
    """
    Agente unificato e 100% Agentico. 
    Usa Qwen per decidere come ingerire l'input e poi lo usa per decomporlo.
    """
    def __init__(self, qwen_instance: Any):
        self.qwen_instance = qwen_instance

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[AGENTE INGESTION & DECOMPOSER] Analizzo l'input grezzo...")
        
        claim_input = state.get("claim_input", {})
        raw_text = claim_input.get("text", "")
        image_path = claim_input.get("image", None)
        
        # ---------------------------------------------------------
        # FASE 1: TOOL SELECTION (L'IA decide la natura dell'input)
        # ---------------------------------------------------------
        testo_log = raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
        task_context = f"The user provided this raw text input: '{testo_log}'." if raw_text else "The user provided no text input."
        if image_path:
             task_context += f" and this image path: '{image_path}'."
        task_context += " Select 'scrape_text_from_url' if the text is a web link. Select 'validate_image' if an image path is provided or if the text itself is an image path. Select 'process_plain_text' if it is a normal medical claim. You can select multiple tools if needed."

        available_tools = ["scrape_text_from_url", "validate_image", "process_plain_text"]
        
        # Qwen decide quali tool usare
        scelte_tool = self.qwen_instance.decide_tool(task_context, available_tools)
        print(f" -> Qwen ha deciso di usare i tool: {scelte_tool}")
        
        # ---------------------------------------------------------
        # FASE 2: TOOL EXECUTION E APPEND COMBINATO
        # ---------------------------------------------------------
        processed_text = raw_text
        final_image_path = None
        
        if not scelte_tool:
            print(" -> ⚠️ Nessun tool richiesto (anomalo). Procedo in modalità testo normale come fallback.")

        for scelta_tool in scelte_tool:
            if scelta_tool == "process_plain_text":
                print(" -> Input riconosciuto come testo normale. Nessuna operazione esterna necessaria.")
                # processed_text rimane raw_text
                
            elif scelta_tool == "scrape_text_from_url":
                print(" -> Eseguo lo scraping dell'URL...")
                processed_text = scrape_text_from_url.invoke({"url": raw_text})
                
            elif scelta_tool == "validate_image":
                print(" -> Preparo l'immagine per l'analisi multimodale...")
                path_to_validate = image_path if image_path else raw_text
                if validate_image.invoke({"image_path": path_to_validate}):
                    final_image_path = path_to_validate 
                    if raw_text == path_to_validate:
                        processed_text = ""   
                else:
                    print(f" ❌ Errore: Immagine non trovata o formato non valido: {path_to_validate}")

        # Fallback di sicurezza: se c'è un'immagine ma Qwen non ha scelto il tool
        if image_path and "validate_image" not in scelte_tool:
            print(" -> Verifico l'immagine allegata fisicamente (Fallback)...")
            if validate_image.invoke({"image_path": image_path}):
                final_image_path = image_path

        # Fallback per Qwen Multimodale: se non c'è testo da processare, forniamo un'istruzione per l'immagine
        if not processed_text and final_image_path:
            processed_text = "Analyze this medical image and extract all relevant medical or scientific claims."
        elif processed_text and final_image_path:
            # Istruzione combinata: Il Decomposer analizza sia l'immagine che il testo (o il testo estratto da URL)
            processed_text = f"Analyze the following text/content along with the provided medical image and extract all relevant medical or scientific claims.\n\nText content:\n{processed_text}"

        # ---------------------------------------------------------
        # FASE 3: DECOMPOSIZIONE
        # ---------------------------------------------------------
        print(" -> Procedo con la scomposizione logica (Decomposition)...")
        res = self.qwen_instance.decompose(text_input=processed_text, image_path=final_image_path)
        
        sub_claims = [sc["claim"] for sc in res.get("sub_claims", [])]
        routing_info = {sc["claim"]: sc.get("routes", ["kb"]) for sc in res.get("sub_claims", [])}
        
        # Aggiorniamo il log dei messaggi includendo il processo decisionale
        tools_usati = ", ".join(scelte_tool) if scelte_tool else "nessuno"
        claims_list = "\n".join([f"  - {sc}" for sc in sub_claims]) if sub_claims else "  - Nessun claim estratto."
        log_message = (
            f"Ho analizzato l'input usando i tool: [{tools_usati}].\n"
            f"Ho completato la scomposizione logica estraendo {len(sub_claims)} sub-claims pronti per il retrieval:\n{claims_list}"
        )
        
        # --- GESTIONE SALVATAGGIO MONGODB E DASHBOARD ---
        updated_claim_input = claim_input.copy()
        if not raw_text:
            if sub_claims:
                updated_claim_input["text"] = "[Estrazione da Immagine] " + " • ".join(sub_claims)
            else:
                updated_claim_input["text"] = "[Nessun claim rilevato nell'immagine]"
        
        if "image" in updated_claim_input:
            updated_claim_input["image"] = None

        return {
            "claim_input": updated_claim_input,
            "sub_claims": sub_claims,
            "routing_info": routing_info,
            "messages": [AIMessage(content=log_message, name="IngestionAndDecomposer")]
        }