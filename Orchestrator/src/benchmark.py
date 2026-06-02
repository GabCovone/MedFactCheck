import requests
import time
import os
import json
import random
import asyncio
import sys
import uuid
from pymongo import MongoClient
try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support
except ImportError:
    print("⚠️ Libreria 'scikit-learn' mancante. Installala con: pip install scikit-learn")
    exit(1)

# ==========================================
# INTEGRAZIONE DIRETTA LANGGRAPH (COLAB MODE)
# ==========================================
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from agent.multi_agent import multi_agent_graph

# ==========================================
# CONFIGURAZIONE
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB_NAME", "medfactcheck")

# ==========================================
# TEST SET (SciFact)
# ==========================================
import os
import json
import random
import tarfile
import urllib.request

def load_scifact_dataset(num_samples=10):
    print(f"⏳ Recupero {num_samples} claim casuali dal dataset SciFact ufficiale...")
    
    # Percorsi per il download e l'estrazione locale
    data_dir = "/content/scifact_dataset"
    tar_path = "/content/scifact_data.tar.gz"
    jsonl_path = os.path.join(data_dir, "data", "claims_dev.jsonl")
    
    # 1. Scarica l'archivio ufficiale dal bucket AWS pubblico CORRETTO
    if not os.path.exists(jsonl_path):
        print("   -> Download dell'archivio ufficiale da AWS S3 in corso (attendere)...")
        # URL pubblico estratto dal codice sorgente originale di AllenAI
        url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
        
        # Aggiungiamo un header User-Agent per evitare eventuali blocchi firewall
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
            out_file.write(response.read())
        
        print("   -> Estrazione dell'archivio...")
        os.makedirs(data_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
            
        # Pulizia del file compresso per liberare RAM/Disco
        if os.path.exists(tar_path):
            os.remove(tar_path)
            
    # 2. Leggi il file jsonl appena estratto
    print("   -> Lettura dei claims di validazione in corso...")
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # Leggiamo tutte le righe valide
        lines = [line.strip() for line in f if line.strip()]
        
    # --- MODIFICA CRITICA: Seed crittografico fisso ---
    # Questo garantisce che l'estrazione casuale dei claim sia sempre identica
    random.seed(42)
    random.shuffle(lines)
    
    # 3. Parsing del dataset (Estrazione claim e vera etichetta)
    for line in lines:
        data = json.loads(line)
        claim_text = data.get("claim", "")
        evidence = data.get("evidence", {})
        
        true_label = "Not Enough Information"
        for doc_id, ev_list in evidence.items():
            for ev in ev_list:
                if ev.get("label") == "SUPPORT":
                    true_label = "Supported"
                elif ev.get("label") == "CONTRADICT":
                    true_label = "Refuted"
        
        dataset.append({"claim": claim_text, "true_label": true_label})
        
        if len(dataset) >= num_samples:
            break
            
    print(f"✅ Recuperati {len(dataset)} claim reali pronti per il benchmark.")
    return dataset

async def process_claim(idx, total, item, collection, semaphore):
    async with semaphore:
        claim = item["claim"]
        true_label = item["true_label"]
        
        print(f"\n[{idx}/{total}] Avvio test in parallelo: '{claim[:50]}...'")
        
        try:
            start_time = time.time()
            
            # Invocazione diretta del grafo (bypassa FastAPI e Timeout HTTP)
            inputs = {
                "claim_input": {"text": claim},
                "claim_id": "",
                "sub_claims": [],
                "routing_info": {},
                "retrieved_docs": {},
                "reasoning_output": {},
                "veracity_results": {},
                "messages": []
            }
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            final_state = await multi_agent_graph.ainvoke(inputs, config=config)
            claim_id = final_state.get("claim_id")
            
            if not claim_id:
                print(f"   ❌ [{idx}] Errore: Nessun claim_id restituito.")
                return true_label, "Error"
                
            # Il server ha concluso. Interroghiamo Mongo per il verdetto finale.
            doc = collection.find_one({"claim_id": claim_id})
            pred_label = doc["final_verdict"] if doc and "final_verdict" in doc else "Not Enough Information"
                    
            elapsed = round(time.time() - start_time, 2)
            print(f"   ✓ [{idx}] Predizione: {pred_label} | Verità: {true_label} | Tempo: {elapsed}s")
            return true_label, pred_label
            
        except Exception as e:
            print(f"   ❌ [{idx}] Errore durante l'esecuzione del grafo: {e}")
            return true_label, "Error"

async def run_benchmark():
    print("==================================================")
    print("🚀 MEDFACTCHECK - BENCHMARK & EVALUATION SCRIPT")
    print("==================================================\n")
    
    # Specifica qui quanti claim vuoi testare
    NUM_SAMPLES = 10
    TEST_DATASET = load_scifact_dataset(num_samples=NUM_SAMPLES)
    
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    collection = db["final_results"]
    
    y_true = []
    y_pred = []
    
    print(f"Inizio valutazione su {len(TEST_DATASET)} claim...\n")
    
    # --- ESECUZIONE PARALLELA TRAMITE LANGGRAPH ---
    # Limitato a 2 per non causare Out of Memory (OOM) sui 15GB di VRAM di Google Colab
    MAX_CONCURRENT_TASKS = 2 
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    tasks = [process_claim(idx, len(TEST_DATASET), item, collection, semaphore) for idx, item in enumerate(TEST_DATASET, 1)]
    results = await asyncio.gather(*tasks)
    
    for true_label, pred_label in results:
        y_true.append(true_label)
        y_pred.append(pred_label)
            
    print("\n==================================================")
    print("📊 RISULTATI DEL BENCHMARK (METRICHE QUANTITATIVE)")
    print("==================================================\n")
    
    labels = ["Supported", "Refuted", "Not Enough Information"]
    
    # Filtriamo eventuali errori di rete dai calcoli finali per non inquinare le metriche
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    print("==================================================")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # Necessario per Colab/Jupyter
    asyncio.run(run_benchmark())