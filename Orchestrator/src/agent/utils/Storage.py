import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

# ── URI di connessione ────────────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
print(f"✅ URI impostata su locale: {MONGO_URI}")

DB_NAME = "medfactcheck"

# ── Test connessione ──────────────────────────────────────────────────────
try:
    _test_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    _test_client.admin.command("ping")
    print(f"✅ Connessione MongoDB OK → database: '{DB_NAME}'")
    _test_client.close()
except ConnectionFailure as e:
    print(f"❌ Connessione fallita: {e}")
    print("   Verifica che MongoDB sia in esecuzione o che l'URI Atlas sia corretta.")


import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from langchain_core.messages import AIMessage


class StorageManager:
    """
    Gestisce la persistenza di tutti i dati di MedFactCheck su MongoDB.

    Collezioni:
      - claims         : claim originale + sub-claims con routes (da ClaimsProcessing)
      - evidence       : passage recuperati per ogni sub-claim (da RetrieverAgent)
      - verdicts       : CoT + label + confidence per ogni sub-claim (da Reasoning_Veracity)
      - final_results  : verdetto aggregato + agent trace (per la Dashboard)
    """

    # ── Inizializzazione ──────────────────────────────────────────────────────

    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME):
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._db     = self._client[db_name]

        self.claims        : Collection = self._db["claims"]
        self.evidence      : Collection = self._db["evidence"]
        self.verdicts      : Collection = self._db["verdicts"]
        self.final_results : Collection = self._db["final_results"]

        self._ensure_indexes()
        print(f"✅ StorageManager inizializzato → DB: '{db_name}'")

    def _ensure_indexes(self):
        """Crea gli indici necessari (idempotente)."""
        # claims
        self.claims.create_index([("claim_id", ASCENDING)], unique=True)
        self.claims.create_index([("timestamp", DESCENDING)])

        # evidence
        self.evidence.create_index([("claim_id", ASCENDING)])
        self.evidence.create_index([("sub_claim_hash", ASCENDING)])
        self.evidence.create_index([("source", ASCENDING)])

        # verdicts
        self.verdicts.create_index([("claim_id", ASCENDING)])
        self.verdicts.create_index([("verdict", ASCENDING)])
        self.verdicts.create_index([("timestamp", DESCENDING)])

        # final_results
        self.final_results.create_index([("claim_id", ASCENDING)], unique=True)
        self.final_results.create_index([("final_verdict", ASCENDING)])
        self.final_results.create_index([("timestamp", DESCENDING)])
        print("   📌 Indici verificati/creati.")

    def close(self):
        self._client.close()

    # ── Utility interna ───────────────────────────────────────────────────────

    @staticmethod
    def _make_claim_id(text: str) -> str:
        """Genera un claim_id deterministico dall'hash del testo."""
        digest = hashlib.md5(text.lower().strip().encode()).hexdigest()[:10]
        return f"claim_{digest}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ═══════════════════════════════════════════════════════════════════════════
    # COLLEZIONE: claims
    # Input: output di ClaimsProcessing (QwenNF4Decomposer.decompose())
    # ═══════════════════════════════════════════════════════════════════════════

    def save_claim(
        self,
        original_text: str,
        source_type: str = "text"
    ) -> str:
        """
        Salva il claim originale nello stato iniziale 'received'.

        Args:
            original_text: testo grezzo del claim
            source_type  : "text" | "url" | "image"

        Returns:
            claim_id (str)
        """
        claim_id  = self._make_claim_id(original_text)
        
        document = {
            "claim_id"      : claim_id,
            "original_text" : original_text,
            "source_type"   : source_type,
            "timestamp"     : self._now(),
            "status"        : "received"
        }
        
        self.claims.update_one(
            {"claim_id": claim_id},
            {"$set": document},
            upsert=True
        )
        print(f"💾 [claims] Inizializzato claim '{claim_id}'")
        return claim_id

    def save_claim_decomposition(
        self,
        claim_id: str,
        decomposer_output: dict
    ) -> str:
        """
        Aggiorna il claim esistente con i risultati della decomposizione.

        Args:
            claim_id         : ID del claim da aggiornare
            decomposer_output: dizionario restituito da QwenNF4Decomposer.decompose()
        """

        # Arricchiamo ogni sub-claim con un hash univoco e metadati
        sub_claims_enriched = []
        for sc in decomposer_output.get("sub_claims", []):
            sub_claims_enriched.append({
                "claim"         : sc["claim"],
                "routes"        : sc.get("routes", []),
                "sub_claim_hash": hashlib.md5(sc["claim"].encode()).hexdigest()[:8],
                "verdict_ready" : False   # diventa True dopo Reasoning_Veracity
            })

        update_data = {
            "decomposer_reasoning": decomposer_output.get("reasoning", ""),
            "sub_claims"    : sub_claims_enriched,
            "n_sub_claims"  : len(sub_claims_enriched),
            "status"        : "decomposed",
            "last_updated"  : self._now()
        }

        self.claims.update_one(
            {"claim_id": claim_id},
            {"$set": update_data}
        )
        print(f"💾 [claims] Aggiornata decomposizione per '{claim_id}' — {len(sub_claims_enriched)} sub-claim/s")
        return claim_id

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Recupera un claim completo per claim_id."""
        return self.claims.find_one({"claim_id": claim_id}, {"_id": 0})

    def update_claim_status(self, claim_id: str, status: str):
        """Aggiorna lo status del claim (decomposed→retrieved→verified→done)."""
        self.claims.update_one(
            {"claim_id": claim_id},
            {"$set": {"status": status, "last_updated": self._now()}}
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # COLLEZIONE: evidence
    # Input: output di RetrieverAgent (retrieve_evidence())
    # ═══════════════════════════════════════════════════════════════════════════

    def save_evidence(self, claim_id: str, retriever_output: dict) -> list[str]:
        """
        Salva i passage di evidenza recuperati dal RetrieverAgent.

        Args:
            claim_id        : ID del claim padre (da save_claim_decomposition)
            retriever_output: dizionario restituito da retrieve_evidence()
                              Struttura attesa:
                              {
                                "claim_id": "...",
                                "claim"   : "testo del sub-claim",
                                "claim_analysis": {...},
                                "evidence_passages": [
                                  {
                                    "source": "PubMed|SciFact|EuropePMC",
                                    "doc_id": "...", "pmid": "...", "doi": "...",
                                    "title": "...", "abstract": "...", "text": "...",
                                    "authors": "...", "journal": "...", "year": "...",
                                    "rrf_score": 0.045,
                                    "relevance_score": 0.9,
                                    "verdict_direction": "supports|refutes|neutral",
                                    "evidence_type": "RCT|meta-analysis|...",
                                    "key_finding": "..."
                                  }, ...
                                ],
                                "stats": {...},
                                "errors": [...]
                              }

        Returns:
            lista di evidence_id inseriti
        """
        sub_claim_text = retriever_output.get("claim", "")
        sub_claim_hash = hashlib.md5(sub_claim_text.encode()).hexdigest()[:8]
        passages       = retriever_output.get("evidence_passages", [])
        evidence_ids   = []

        docs_to_insert = []
        for p in passages:
            # Il retriever ora fornisce 'text', 'source' e 'score', non 'doc_id'. Usiamo l'hash del testo.
            passage_text = p.get("text") or p.get("abstract") or p.get("testo", "")
            passage_hash = hashlib.md5(passage_text.encode()).hexdigest()[:8]
            evidence_id = f"ev_{claim_id}_{sub_claim_hash}_{passage_hash}"
            
            doc = {
                "evidence_id"       : evidence_id,
                "claim_id"          : claim_id,
                "sub_claim"         : sub_claim_text,
                "sub_claim_hash"    : sub_claim_hash,
                "claim_analysis"    : retriever_output.get("claim_analysis", {}),
                # — campi passage —
                "source"            : p.get("source", ""),
                "doc_id"            : p.get("doc_id", ""),
                "pmid"              : p.get("pmid", ""),
                "doi"               : p.get("doi", ""),
                "pmcid"             : p.get("pmcid", ""),
                "title"             : p.get("title", ""),
                "testo"             : passage_text,
                "authors"           : p.get("authors", ""),
                "journal"           : p.get("journal", ""),
                "year"              : p.get("year", ""),
                "has_fulltext"      : p.get("has_fulltext", False),
                "url"               : p.get("url", ""),
                # — score di retrieval —
                "rrf_score"         : p.get("rrf_score", 0.0),
                "relevance_score"   : p.get("score", p.get("relevance_score", None)),
                "verdict_direction" : p.get("verdict_direction", "neutral"),
                "evidence_type"     : p.get("evidence_type", "other"),
                "key_finding"       : p.get("key_finding", ""),
                # — metadati —
                "retrieval_stats"   : retriever_output.get("stats", {}),
                "retriever_errors"  : retriever_output.get("errors", []),
                "timestamp"         : self._now()
            }
            docs_to_insert.append(doc)
            evidence_ids.append(evidence_id)

        if docs_to_insert:
            for doc in docs_to_insert:
                self.evidence.update_one(
                    {"evidence_id": doc["evidence_id"]},
                    {"$set": doc},
                    upsert=True
                )
            print(f"💾 [evidence] {len(docs_to_insert)} passage salvati per '{claim_id}' (sub: {sub_claim_hash})")
        else:
            print(f"⚠️  [evidence] Nessun passage da salvare per '{claim_id}'")

        return evidence_ids

    def get_evidence_for_claim(self, claim_id: str) -> list[dict]:
        """Recupera tutti i passage di evidenza per un claim_id."""
        return list(self.evidence.find(
            {"claim_id": claim_id},
            {"_id": 0}
        ).sort("relevance_score", DESCENDING))

    def get_evidence_for_sub_claim(self, claim_id: str, sub_claim_hash: str) -> list[dict]:
        """Recupera i passage relativi a uno specifico sub-claim."""
        return list(self.evidence.find(
            {"claim_id": claim_id, "sub_claim_hash": sub_claim_hash},
            {"_id": 0}
        ).sort("rrf_score", DESCENDING))

    # ═══════════════════════════════════════════════════════════════════════════
    # COLLEZIONE: verdicts
    # Input: output di Reasoning_Veracity (ReasoningAndVeracityPipeline.process_claim())
    # ═══════════════════════════════════════════════════════════════════════════

    def save_verdict(self, claim_id: str, veracity_output: dict) -> str:
        """
        Salva il verdetto (CoT + label + confidence) di un sub-claim.

        Args:
            claim_id       : ID del claim padre
            veracity_output: dizionario restituito da process_claim()
                             Struttura attesa:
                             {
                               "claim"              : "testo del sub-claim",
                               "verdict"            : "Supported|Refuted|Not Enough Information",
                               "confidence_score"   : 0.91,
                               "chain_of_thought_log": "...",
                               "supporting_evidence": [
                                 {"titolo": "...", "url": "...", "testo": "..."}
                               ]
                             }

        Returns:
            verdict_id (str)
        """
        sub_claim_text = veracity_output.get("claim", "")
        sub_claim_hash = hashlib.md5(sub_claim_text.encode()).hexdigest()[:8]
        verdict_id     = f"vrd_{claim_id}_{sub_claim_hash}"

        document = {
            "verdict_id"          : verdict_id,
            "claim_id"            : claim_id,
            "sub_claim"           : sub_claim_text,
            "sub_claim_hash"      : sub_claim_hash,
            "verdict"             : veracity_output.get("verdict", "Not Enough Information"),
            "confidence_score"    : veracity_output.get("confidence_score", 0.0),
            "chain_of_thought_log": veracity_output.get("chain_of_thought_log", ""),
            "supporting_evidence" : veracity_output.get("supporting_evidence", []),
            "timestamp"           : self._now()
        }

        self.verdicts.update_one(
            {"verdict_id": verdict_id},
            {"$set": document},
            upsert=True
        )

        # Aggiorna il flag verdict_ready nel documento claims
        self.claims.update_one(
            {"claim_id": claim_id, "sub_claims.sub_claim_hash": sub_claim_hash},
            {"$set": {"sub_claims.$.verdict_ready": True}}
        )

        print(f"💾 [verdicts] '{verdict_id}' → {document['verdict']} ({document['confidence_score']:.2%})")
        return verdict_id

    def get_verdicts_for_claim(self, claim_id: str) -> list[dict]:
        """Recupera tutti i verdetti per un claim_id."""
        return list(self.verdicts.find(
            {"claim_id": claim_id},
            {"_id": 0}
        ).sort("timestamp", ASCENDING))

    # ═══════════════════════════════════════════════════════════════════════════
    # COLLEZIONE: final_results
    # Aggregazione per la Dashboard — chiamata dal Coordinator Agent
    # ═══════════════════════════════════════════════════════════════════════════

    def aggregate_final_verdict(self, claim_id: str, agent_trace: list = None) -> dict:
        """
        Aggrega i verdetti dei sub-claim in un verdetto finale e lo salva.

        Logica di aggregazione:
          - Almeno 1 sub-claim Refuted  → finale = Refuted
          - Tutti Supported             → finale = Supported
          - Altrimenti                  → finale = Not Enough Information

        Args:
            claim_id   : ID del claim
            agent_trace: lista opzionale di step dell'agent (per la dashboard)

        Returns:
            documento final_result completo
        """
        claim_doc    = self.get_claim(claim_id)
        sub_verdicts = self.get_verdicts_for_claim(claim_id)

        if not sub_verdicts:
            print(f"⚠️  [final_results] Nessun verdetto trovato per '{claim_id}'")
            return {}

        labels      = [v["verdict"] for v in sub_verdicts]
        avg_conf    = round(sum(v["confidence_score"] for v in sub_verdicts) / len(sub_verdicts), 4)

        # Regola di aggregazione
        if "Refuted" in labels:
            final_verdict = "Refuted"
        elif all(l == "Supported" for l in labels):
            final_verdict = "Supported"
        else:
            final_verdict = "Not Enough Information"

        document = {
            "claim_id"          : claim_id,
            "original_text"     : claim_doc.get("original_text", "") if claim_doc else "",
            "final_verdict"     : final_verdict,
            "avg_confidence"    : avg_conf,
            "n_sub_claims"      : len(sub_verdicts),
            "verdict_breakdown" : {
                "Supported"             : labels.count("Supported"),
                "Refuted"               : labels.count("Refuted"),
                "Not Enough Information": labels.count("Not Enough Information")
            },
            "sub_verdicts"      : sub_verdicts,
            "agent_trace"       : agent_trace or [],
            "timestamp"         : self._now()
        }

        self.final_results.update_one(
            {"claim_id": claim_id},
            {"$set": document},
            upsert=True
        )

        # Aggiorna status nel documento claims
        self.update_claim_status(claim_id, "done")

        print(f"🏁 [final_results] '{claim_id}' → VERDETTO FINALE: {final_verdict} (conf media: {avg_conf:.2%})")
        return document

    # ═══════════════════════════════════════════════════════════════════════════
    # METODI DI QUERY — per la Dashboard Streamlit
    # ═══════════════════════════════════════════════════════════════════════════

    def get_final_result(self, claim_id: str) -> Optional[dict]:
        """Recupera il risultato finale completo per claim_id."""
        return self.final_results.find_one({"claim_id": claim_id}, {"_id": 0})

    def get_all_results(
        self,
        verdict_filter: str = None,
        limit: int = 50,
        skip: int = 0
    ) -> list[dict]:
        """
        Recupera i risultati finali per la dashboard (con filtri opzionali).

        Args:
            verdict_filter: "Supported" | "Refuted" | "Not Enough Information" | None
            limit         : numero massimo di risultati
            skip          : offset per paginazione
        """
        query = {}
        if verdict_filter:
            query["final_verdict"] = verdict_filter

        return list(
            self.final_results
            .find(query, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .skip(skip)
            .limit(limit)
        )

    def search_claims(self, keyword: str, limit: int = 20) -> list[dict]:
        """Ricerca full-text nel testo originale dei claim."""
        self.final_results.create_index([("original_text", "text")], background=True)
        return list(
            self.final_results
            .find(
                {"$text": {"$search": keyword}},
                {"_id": 0, "score": {"$meta": "textScore"}}
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

    def get_stats(self) -> dict:
        """Statistiche globali per la dashboard."""
        total  = self.final_results.count_documents({})
        if total == 0:
            return {"total": 0, "supported": 0, "refuted": 0, "nei": 0, "avg_confidence": 0.0}

        pipeline = [
            {"$group": {
                "_id"           : "$final_verdict",
                "count"         : {"$sum": 1},
                "avg_confidence": {"$avg": "$avg_confidence"}
            }}
        ]
        agg = {r["_id"]: r for r in self.final_results.aggregate(pipeline)}

        return {
            "total"          : total,
            "supported"      : agg.get("Supported",              {}).get("count", 0),
            "refuted"        : agg.get("Refuted",                {}).get("count", 0),
            "nei"            : agg.get("Not Enough Information", {}).get("count", 0),
            "avg_confidence" : round(
                sum(r.get("avg_confidence", 0.0) for r in agg.values()) / len(agg), 4
            ) if agg else 0.0,
            "sources_used"   : list(self.evidence.distinct("source"))
        }

    def delete_claim(self, claim_id: str) -> dict:
        """Elimina un claim e tutti i dati associati (cascade delete)."""
        r1 = self.claims.delete_many({"claim_id": claim_id})
        r2 = self.evidence.delete_many({"claim_id": claim_id})
        r3 = self.verdicts.delete_many({"claim_id": claim_id})
        r4 = self.final_results.delete_many({"claim_id": claim_id})
        summary = {
            "claims_deleted"  : r1.deleted_count,
            "evidence_deleted": r2.deleted_count,
            "verdicts_deleted": r3.deleted_count,
            "results_deleted" : r4.deleted_count
        }
        print(f"🗑️  Eliminati tutti i dati per '{claim_id}': {summary}")
        return summary


print("✅ Classe StorageManager definita.")


async def init_db(state: Dict[str, Any]) -> Dict[str, Any]:
    """Inizializza e verifica la connessione al database MongoDB."""
    print("--- INIT DATABASE ---")
    try:
        storage = StorageManager()
        storage.close()
        print("✅ Connessione al database MongoDB verificata con successo.")
        return {"db_initialized": True}
    except Exception as e:
        print(f"❌ Errore di connessione al database MongoDB: {e}")
        return {"db_initialized": False}