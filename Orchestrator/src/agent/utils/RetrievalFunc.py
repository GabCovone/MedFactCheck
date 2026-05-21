from typing import Any, Dict


async def init_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    model = "BM-25, BioBERT"
    return {"retrieval_models": model}