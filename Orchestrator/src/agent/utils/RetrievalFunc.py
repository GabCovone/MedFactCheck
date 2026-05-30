from typing import Any, Dict
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor, as_completed

class RetrieverAgent:
    """
    Agente Retriever ottimizzato (Guided Execution Multi-Tool).
    Usa il routing pre-calcolato dal Decomposer per lanciare combinazioni di strumenti.
    """
    def __init__(self, decision_model: Any, retrieval_models: Any):
        self.qwen_instance = decision_model
        self.retriever_instance = retrieval_models

        @tool
        def kb_bm25(query: str) -> list:
            """Recupera definizioni statiche dalla Knowledge Base locale tramite BM25."""
            return self.retriever_instance.kb_node.search_bm25(query, top_k=3)

        @tool
        def lit_europe_pmc(query: str) -> list:
            """Cerca in letteratura studi clinici e articoli scientifici (Europe PMC)."""
            return self.retriever_instance.lit_node.retrieve_and_rerank_massive(query)

        @tool
        def kb_faiss(query: str) -> list:
            """Cerca concetti affini o sinonimi nella Knowledge Base tramite ricerca vettoriale FAISS."""
            return self.retriever_instance.kb_node.search_faiss_pq(query, top_k=3)

        self.available_tools = {
            "kb_bm25": kb_bm25,
            "lit_europe_pmc": lit_europe_pmc,
            "kb_faiss": kb_faiss
        }

    def _route_to_tools(self, routes: list) -> list:
        """
        Logica deterministica per mappare la rotta su PIÙ tools in contemporanea.
        Restituisce una lista di tool da eseguire.
        """
        if routes == ["kb"]:
            return ["kb_bm25"]
        elif "lit" in routes:
            # L'aggiunta richiesta: esegue SIA la ricerca semantica locale SIA la letteratura
            return ["kb_faiss", "lit_europe_pmc"]
        else:
            return ["kb_faiss"]

    def _retrieve_for_one_claim(self, sub_claim: str, routes: list) -> tuple[str, list]:
        """
        Esegue l'intero processo di retrieval per un singolo sub-claim.
        Questa funzione è progettata per essere eseguita in un thread separato.
        """
        tool_names = self._route_to_tools(routes)
        print(f" -> [Thread] Avvio retrieval per '{sub_claim[:40]}...'. Tools: {', '.join(tool_names)}")
        
        docs = []
        # Eseguiamo tutti i tool richiesti e accumuliamo i risultati
        for t_name in tool_names:
            if t_name in self.available_tools:
                try:
                    tool_results = self.available_tools[t_name].invoke({"query": sub_claim})
                    docs.extend(tool_results)
                except Exception as e:
                    print(f"    - ❌ Errore durante l'esecuzione del tool {t_name} per '{sub_claim[:40]}...': {e}")
        
        # GESTIONE FALLIMENTI (Fallback)
        if not docs:
            print(f" ⚠️ Nessun risultato per '{sub_claim[:40]}...'. Attivo Fallback Globale.")
            try:
                docs = self.retriever_instance.retrieve(sub_claim, ["kb", "lit"])
            except Exception as e:
                print(f"    - ❌ Errore durante il Fallback Globale per '{sub_claim[:40]}...': {e}")
            
        print(f"    -> [Thread] Trovati {len(docs)} documenti per '{sub_claim[:40]}...'.")
        return sub_claim, docs

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[AGENTE RETRIEVER] Avvio ricerca guidata parallela per tutti i sub-claim.")
        sub_claims = state.get("sub_claims", [])
        routing_info = state.get("routing_info", {})
        
        retrieved_docs = {}
        
        with ThreadPoolExecutor(max_workers=len(sub_claims) if sub_claims else 1) as executor:
            future_to_sc = {
                executor.submit(self._retrieve_for_one_claim, sc, routing_info.get(sc, ["kb"])): sc 
                for sc in sub_claims
            }
            
            for future in as_completed(future_to_sc):
                try:
                    sc, docs = future.result()
                    retrieved_docs[sc] = docs
                except Exception as e:
                    sc = future_to_sc[future]
                    print(f"❌ Errore critico nel thread di retrieval per '{sc}': {e}")
                    retrieved_docs[sc] = []

        print(f"\n✅ Ricerca parallela completata. Recuperate evidenze per {len(retrieved_docs)}/{len(sub_claims)} sub-claims.")
            
        log_details = "\n".join([f"  - '{sc[:60]}...': trovati {len(docs)} documenti." for sc, docs in retrieved_docs.items()])
        log_message = f"Ricerca ibrida (KB/LIT) completata per {len(sub_claims)} sub-claims. Dettagli:\n{log_details}"
        return {
            "retrieved_docs": retrieved_docs,
            "messages": [AIMessage(content=log_message, name="Retriever")]
        }