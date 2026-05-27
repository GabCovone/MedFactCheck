from typing import Any, Dict
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

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

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[AGENTE RETRIEVER] Avvio ricerca guidata (Guided Execution Multi-Tool).")
        sub_claims = state.get("sub_claims", [])
        routing_info = state.get("routing_info", {})
        
        retrieved_docs = {}
        for sc in sub_claims:
            rotte_suggerite = routing_info.get(sc, ["kb"])
            tool_names = self._route_to_tools(rotte_suggerite)
            
            print(f" -> [ROUTING] Rotta '{rotte_suggerite}'. Eseguo i tools: {', '.join(tool_names)}")
            
            docs = []
            # Eseguiamo tutti i tool richiesti e accumuliamo i risultati
            for t_name in tool_names:
                if t_name in self.available_tools:
                    print(f"    - Lancio {t_name}...")
                    tool_results = self.available_tools[t_name].invoke({"query": sc})
                    docs.extend(tool_results)
            
            # GESTIONE FALLIMENTI (Fallback)
            if not docs:
                print(f" ⚠️ Nessun risultato utile. Attivo Fallback Globale su tutti i rami...")
                docs = self.retriever_instance.retrieve(sc, ["kb", "lit"])
                
            retrieved_docs[sc] = docs
            print(f"    ✅ Trovati {len(docs)} documenti combinati per questo sub-claim.")
            
        return {
            "retrieved_docs": retrieved_docs,
            "messages": [AIMessage(content=f"Ricerca completata per {len(sub_claims)} sub-claims.", name="Retriever")]
        }