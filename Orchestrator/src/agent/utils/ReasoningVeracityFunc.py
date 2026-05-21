from typing import Any, Dict


async def init_veracity(state: Dict[str, Any]) -> Dict[str, Any]:
    model = "BioBERTa"
    return {"veracity_model": model}

async def init_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
    model = "Qwen"
    return {"reasoning_model": model}
