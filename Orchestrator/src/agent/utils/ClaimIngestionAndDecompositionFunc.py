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
        
        # ---------------------------------------------------------
        # FASE 1: TOOL SELECTION (L'IA decide la natura dell'input)
        # ---------------------------------------------------------
        task_context = f"L'utente ha fornito questo input grezzo: '{raw_text[:100]}...'. Devi capire se è un link a un sito web, un percorso a un file immagine o un testo normale."
        available_tools = ["scrape_text_from_url", "validate_image"]
        
        # Qwen decide quale tool usare
        scelta_tool = self.qwen_instance.decide_tool(task_context, available_tools)
        print(f" -> Qwen ha deciso di usare il tool: {scelta_tool}")
        
        # ---------------------------------------------------------
        # FASE 2: TOOL EXECUTION (Esecuzione deterministica)
        # ---------------------------------------------------------
        processed_text = raw_text
        image_path = claim_input.get("image", None)
        
        if scelta_tool == "scrape_text_from_url":
            print(" -> Eseguo lo scraping dell'URL...")
            # Usiamo il tool che avevi già definito
            processed_text = scrape_text_from_url.invoke({"url": raw_text})
            
        elif scelta_tool == "validate_image":
            print(" -> Preparo l'immagine per l'analisi multimodale...")
            # ORA CHIAMIAMO IL TOOL PER VERIFICARE
            if validate_image.invoke({"image_path": raw_text}):
                image_path = raw_text 
                processed_text = ""   
            else:
                print(" ❌ Errore: Immagine non trovata o formato non valido. Fallback a testo.")
                scelta_tool = "nessuno"
                processed_text = raw_text
            
        elif scelta_tool == "nessuno":
            print(" -> Input riconosciuto come testo normale.")
            
        else:
            print(" ⚠️ Tool non riconosciuto, procedo in modalità testo normale come fallback.")

        # ---------------------------------------------------------
        # FASE 3: DECOMPOSIZIONE
        # ---------------------------------------------------------
        print(" -> Procedo con la scomposizione logica (Decomposition)...")
        res = self.qwen_instance.decompose(text_input=processed_text, image_path=image_path)
        
        sub_claims = [sc["claim"] for sc in res.get("sub_claims", [])]
        routing_info = {sc["claim"]: sc.get("routes", ["kb"]) for sc in res.get("sub_claims", [])}
        
        # Aggiorniamo il log dei messaggi includendo il processo decisionale
        log_message = (
            f"Ho analizzato l'input usando il tool '{scelta_tool}'. "
            f"Successivamente ho estratto {len(sub_claims)} sub-claims pronti per il retrieval."
        )
        
        return {
            "sub_claims": sub_claims,
            "routing_info": routing_info,
            "messages": [AIMessage(content=log_message, name="IngestionAndDecomposer")]
        }