import os
import gc
import json
import time
import pickle
import faiss
import torch
import numpy as np
import requests
import torch.nn.functional as F
import gzip
import io
import csv
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import Any, Dict
from bs4 import BeautifulSoup

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. RAMO KB (Inizializzatore e Retriever)
# ==========================================
class KBBranchInitializer:
    def __init__(self, shared_embedder, cache_dir="/content/drive/MyDrive/MedFactCheck_Cache/"):
        self.cache_dir = cache_dir
        self.embedder = shared_embedder
        os.makedirs(self.cache_dir, exist_ok=True)

        self.texts_paths = [os.path.join(self.cache_dir, f"disgenet_extended_texts_v{i}.json") for i in range(1, 4)]
        self.bm25_paths = [os.path.join(self.cache_dir, f"disgenet_extended_bm25_v{i}.pkl") for i in range(1, 4)]
        self.faiss_paths = [os.path.join(self.cache_dir, f"disgenet_extended_faiss_v{i}.index") for i in range(1, 4)]

        self.doc_texts, self.bm25, self.faiss_index = None, None, None
        self._initialize_all()

    def _initialize_all(self):
        # 1. TESTI (Download Bulletproof)
        self.doc_texts = self._load_with_fallback(self.texts_paths, json.load, "r")
        if not self.doc_texts:
            print("-> [KB] Lettura DisGeNET (Curated Gene-Disease Associations) da locale...")
            utils_dir = os.path.dirname(os.path.abspath(__file__))
            disgenet_dir = os.path.join(utils_dir, "Disgenet")
            local_gene_path = os.path.join(disgenet_dir, "curated_gene_disease_associations.tsv")

            self.doc_texts = []
            
            if not os.path.exists(local_gene_path):
                raise RuntimeError(f"Errore Critico: File non trovato in {local_gene_path}. Assicurati di aver inserito i file nella cartella Disgenet.")
            
            print(f"   - Trovato file locale: {local_gene_path}")
            with open(local_gene_path, 'rt', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    gene = row.get('geneSymbol', '')
                    disease = row.get('diseaseName', '')
                    score = row.get('score', '')
                    if gene and disease:
                        self.doc_texts.append(f"The gene {gene} is associated with the disease {disease} with a score of {score}.")
                        
            if not self.doc_texts:
                raise RuntimeError("Errore critico: Nessun record estratto da DisGeNET. Interruzione.")

            self._save_triple_backup(self.doc_texts, self.texts_paths, lambda d, f: json.dump(d, f), "w")

        # 2. BM-25
        self.bm25 = self._load_bm25_with_fallback(len(self.doc_texts))
        if not self.bm25:
            print("-> [KB] Creazione indice BM25...")
            self.bm25 = BM25Okapi([doc.lower().split() for doc in self.doc_texts])
            self._save_triple_backup(self.bm25, self.bm25_paths, lambda d, f: pickle.dump(d, f), "wb")

        # 3. FAISS (PQ)
        self.faiss_index = self._load_faiss_with_fallback(len(self.doc_texts))
        if not self.faiss_index:
            print("-> [KB] Addestramento FAISS PQ in corso...")
            embeddings = self.embedder.encode(self.doc_texts, convert_to_numpy=True, show_progress_bar=True)

            embeddings = embeddings.astype('float32')

            faiss.normalize_L2(embeddings)
            d, m, nbits = embeddings.shape[1], 96, 8
            self.faiss_index = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.train(embeddings)
            self.faiss_index.add(embeddings)
            for path in self.faiss_paths:
                faiss.write_index(self.faiss_index, path)
            del embeddings
            gc.collect()
            torch.cuda.empty_cache()

    def _load_with_fallback(self, paths, load_func, mode):
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, mode) as f: return load_func(f)
                except Exception: pass
        return None

    def _load_bm25_with_fallback(self, expected_size):
        for path in self.bm25_paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        bm25_idx = pickle.load(f)
                        if bm25_idx.corpus_size == expected_size:
                            print(f"-> [KB] Indice BM25 integro caricato da {path}")
                            return bm25_idx
                        else:
                            print(f"-> [KB] Avviso: Indice BM25 in {path} obsoleto. Provo backup...")
                except Exception as e:
                    print(f"-> [KB] Errore lettura BM25: {e}")
        return None

    def _load_faiss_with_fallback(self, expected_size):
        for path in self.faiss_paths:
            if os.path.exists(path):
                try: 
                    idx = faiss.read_index(path)
                    if idx.ntotal == expected_size:
                        print(f"-> [KB] Indice FAISS integro caricato da {path}")
                        return idx
                    else:
                        print(f"-> [KB] Avviso: Indice FAISS in {path} obsoleto. Provo backup...")
                except Exception as e: 
                    print(f"-> [KB] Errore lettura FAISS da {path}: {e}")
        return None

    def _save_triple_backup(self, data, paths, save_func, mode):
        for path in paths:
            with open(path, mode) as f: save_func(data, f)

class KBRetrieverNode:
    def __init__(self, kb_data, shared_embedder):
        self.doc_texts = kb_data.doc_texts
        self.bm25 = kb_data.bm25
        self.faiss_index = kb_data.faiss_index
        self.embedder = shared_embedder

    def _apply_hard_truncation(self, text: str, max_chars: int = 500) -> str:
        if len(text) <= max_chars: return text
        return text[:max_chars].rsplit(' ', 1)[0] + "..."

    def search_bm25(self, query: str, top_k: int) -> list:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"text": self._apply_hard_truncation(self.doc_texts[idx]),
                 "source": "KB (DisGeNET - BM25)",
                 "score": round(float(scores[idx]), 4)}
                for idx in top_indices if scores[idx] > 0]

    def search_faiss_pq(self, query: str, top_k: int) -> list:
        query_vector = self.embedder.encode([query], convert_to_numpy=True)

        # FIX: Convertiamo il vettore della query in FP32 per FAISS
        query_vector = query_vector.astype('float32')

        faiss.normalize_L2(query_vector)
        distances, indices = self.faiss_index.search(query_vector, top_k)
        return [{"text": self._apply_hard_truncation(self.doc_texts[idx]),
                 "source": "KB (DisGeNET - FAISS)",
                 "score": round(float(dist), 4)}
                for dist, idx in zip(distances[0], indices[0]) if idx != -1]


# ==========================================
# 2. RAMO LIT (Europe PMC Massivo)
# ==========================================
class LitBulkRetrieverNode:
    def __init__(self, shared_embedder):
        self.embedder = shared_embedder
        self.pmc_api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=5000)
        self.papers_collection = self.mongo_client.get_database(os.getenv("MONGO_DB_NAME", "medfactcheck")).get_collection("papers")

    def _fetch_top_papers_metadata(self, query: str, limit: int = 50) -> list:
        params = {"query": f'({query}) AND OPEN_ACCESS:y', "format": "json", "resultType": "lite", "pageSize": limit}
        try:
            res = requests.get(self.pmc_api_url, params=params, timeout=10)
            return [{"id": r.get("pmcid") or r.get("pmid"), "title": r.get("title", "")}
                    for r in res.json().get("resultList", {}).get("result", []) if r.get("pmcid") or r.get("pmid")]
        except Exception: return []

    def _fetch_full_text_from_api(self, pmcid: str) -> str:
        """Scarica il VERO Full-Text XML da Europe PMC e ne estrae il testo pulito."""
        # Endpoint specifico per il full text XML
        full_text_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        try:
            res = requests.get(full_text_url, timeout=10)
            if res.status_code != 200: 
                return ""
            
            # Usiamo BeautifulSoup per parsare l'XML e prendere tutto il testo dei paragrafi
            soup = BeautifulSoup(res.content, "xml")
            
            # Europe PMC racchiude il corpo dell'articolo nel tag <body>
            body = soup.find("body")
            if not body:
                return ""
            
            # Estraiamo tutto il testo rimuovendo i tag XML
            full_text = body.get_text(separator=' ', strip=True)
            return full_text
        except Exception as e: 
            print(f"Errore download full-text per {pmcid}: {e}")
            return ""

    def retrieve_and_rerank_massive(self, claim: str, top_k_bm25: int = 100, top_n_biobert: int = 20) -> list:
        metadata_list = self._fetch_top_papers_metadata(claim, limit=50)
        if not metadata_list: return []

        target_ids = [m["id"] for m in metadata_list]
        cached_docs = list(self.papers_collection.find({"id": {"$in": target_ids}}, {"_id": 0}))
        cached_papers = {doc["id"]: doc for doc in cached_docs}

        final_papers = []
        for meta in metadata_list:
            pid = meta["id"]
            if pid in cached_papers:
                final_papers.append(cached_papers[pid])
            else:
                full_text = self._fetch_full_text_from_api(pid)
                if full_text:
                    paper_doc = {"id": pid, "title": meta["title"], "text": full_text}
                    final_papers.append(paper_doc)
                    self.papers_collection.update_one({"id": pid}, {"$set": paper_doc}, upsert=True)
                time.sleep(0.1) # Pausa di sicurezza

        all_chunks = []
        for p in final_papers:
            words = f"{p['title']}. {p['text']}".split()
            for i in range(0, len(words), 120):
                all_chunks.append({"text": " ".join(words[i:i + 150]), "source": f"PMC ID: {p['id']}"})

        if not all_chunks: return []

        bm25_scores = BM25Okapi([c["text"].lower().split() for c in all_chunks]).get_scores(claim.lower().split())
        candidate_chunks = [all_chunks[i] for i in np.argsort(bm25_scores)[::-1][:min(top_k_bm25, len(all_chunks))] if bm25_scores[i] > 0]

        if not candidate_chunks: return []

        with torch.no_grad():
            claim_emb = self.embedder.encode(claim, convert_to_tensor=True)
            chunk_embs = self.embedder.encode([c["text"] for c in candidate_chunks], convert_to_tensor=True)
            similarities = F.cosine_similarity(claim_emb.unsqueeze(0), chunk_embs)

        top_indices = torch.argsort(similarities, descending=True)[:min(top_n_biobert, len(candidate_chunks))]
        res = [{"text": candidate_chunks[i.item()]["text"], "source": candidate_chunks[i.item()]["source"], "score": round(float(similarities[i.item()]), 4)} for i in top_indices]

        torch.cuda.empty_cache()
        return res


# ==========================================
# 3. L'ORCHESTRATORE GLOBALE
# ==========================================
class MedFactCheckRetriever:
    def __init__(self):
        print(f"[*] Inizializzazione Globale Sistema RAG su: {DEVICE}")
        print("[*] Allocazione S-PubMedBert (FP16) in VRAM...")
        self.embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', model_kwargs={"torch_dtype": torch.float16}, device=DEVICE)

        print("[*] Sincronizzazione Ramo KB (DisGeNET)...")
        kb_data = KBBranchInitializer(shared_embedder=self.embedder)
        self.kb_node = KBRetrieverNode(kb_data, shared_embedder=self.embedder)

        print("[*] Sincronizzazione Ramo LIT (Europe PMC)...")
        self.lit_node = LitBulkRetrieverNode(shared_embedder=self.embedder)

        print("\n✅ SISTEMA DI RETRIEVAL PRONTO E ALLINEATO.")
        gc.collect()
        torch.cuda.empty_cache()

    def retrieve(self, claim: str, routes: list) -> list:
        final_evidence = []
        if routes == ["kb"]:
            print(f" -> [ROTTA KB ESATTA] Esecuzione BM-25 su DisGeNET...")
            final_evidence.extend(self.kb_node.search_bm25(claim, top_k=3))
        elif "lit" in routes:
            print(f" -> [ROTTA COMPLESSA] 1. FAISS su DisGeNET...")
            final_evidence.extend(self.kb_node.search_faiss_pq(claim, top_k=3))
            print(f" -> [ROTTA COMPLESSA] 2. Ricerca Massiva su Europe PMC...")
            final_evidence.extend(self.lit_node.retrieve_and_rerank_massive(claim))
        return final_evidence
