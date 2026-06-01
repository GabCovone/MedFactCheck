from fastapi import FastAPI, Form, UploadFile, File, HTTPException
import os
import sys
import tempfile
import uuid
import base64

# Configura il path in modo che trovi il modulo 'agent'
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from agent.multi_agent import multi_agent_graph

app = FastAPI(
    title="MedFactCheck API",
    description="API asincrona per la verifica di claim medici tramite sistema Multi-Agente",
    version="1.0.0"
)

@app.post("/verify")
async def verify_claim(text: str = Form(None), image: UploadFile = File(None)):
    """
    Endpoint per verificare un claim medico. 
    Riceve testo e/o un'immagine, avvia la pipeline e restituisce l'ID.
    """
    if not text and not image:
        raise HTTPException(status_code=400, detail="Devi fornire almeno un testo o un'immagine per la verifica.")

    image_path = None
    image_b64 = None
    try:
        # Se è stata fornita un'immagine, la salviamo temporaneamente sul server
        if image and image.filename:
            image_bytes = await image.read()
            file_ext = image.filename.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(image_bytes)
                image_path = tmp_file.name
                
            b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            mime_type = "image/jpeg" if file_ext.lower() in ["jpg", "jpeg"] else f"image/{file_ext.lower()}"
            image_b64 = f"data:{mime_type};base64,{b64_encoded}"

        # Prepariamo lo State per LangGraph
        inputs = {
            "claim_input": {},
            "claim_id": "",
            "sub_claims": [],
            "routing_info": {},
            "retrieved_docs": {},
            "reasoning_output": {},
            "veracity_results": {},
            "messages": []
        }
        if text:
            inputs["claim_input"]["text"] = text
        if image_path:
            inputs["claim_input"]["image"] = image_path
            inputs["claim_input"]["image_b64"] = image_b64

        # Invocazione asincrona del Grafo
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        log_text = f"'{text[:50]}...'" if text else "[Solo Immagine]"
        print(f"\n[API] Ricevuta richiesta di verifica per: {log_text}")
        final_state = await multi_agent_graph.ainvoke(inputs, config=config)

        # Restituiamo il claim_id al client in modo che possa interrogare MongoDB per i dettagli
        return {"claim_id": final_state.get("claim_id")}

    except Exception as e:
        print(f"[API] Errore critico: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Pulizia rigorosa del file temporaneo lato server
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

@app.get("/health")
async def health_check():
    """Endpoint di controllo stato del server."""
    return {"status": "ok", "message": "MedFactCheck API is running"}