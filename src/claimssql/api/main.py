"""HTTP API for the abstention-aware Text-to-SQL pipeline (same pipeline as the app).

    GET  /health   is the service ready: database, model, API key
    GET  /schema   the tables and columns the model is shown
    POST /ask      {"question": "..."} -> an answer or an abstention

Run locally:
    uvicorn claimssql.api.main:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from claimssql.api.pipeline import DemoService
from claimssql.llm.base import LLMError, RateLimitStop

app = FastAPI(title="ClaimsSQL-Trust", version="0.1.0",
              description="Text-to-SQL over CMS DE-SynPUF claims that can say "
                          "'I can't answer that from this database'.")  # fmt: skip


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def get_service() -> DemoService:
    return DemoService.from_env()


# One shared service per process (database, model client, catalogue).
Service = Annotated[DemoService, Depends(get_service)]


@app.get("/health")
def health(service: Service) -> dict[str, Any]:
    cursor = service.con.cursor()  # a cursor per request: connections are not thread-safe
    try:
        cursor.execute("SELECT 1").fetchone()  # the database answers
    finally:
        cursor.close()
    return {
        "status": "ok",
        "database": str(service.db_path) if service.db_path else None,
        "model": service.client.model,
        "api_key_set": bool(os.getenv("GROQ_API_KEY")),
    }


@app.get("/schema")
def schema(service: Service) -> dict[str, Any]:
    tables = [
        {
            "name": table.name,
            "grain": table.grain,
            "description": table.description,
            "columns": [
                {"name": c.name, "type": c.type, "description": c.description, "codes": c.codes}
                for c in table.columns if c.include
            ],
        }
        for table in service.catalogue.tables
    ]  # fmt: skip
    return {"tables": tables, "context": service.context}


@app.post("/ask")
def ask(request: AskRequest, service: Service) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")
    try:
        return service.ask(question).to_dict()
    except RateLimitStop as exc:
        raise HTTPException(
            status_code=429,
            detail=f"The model provider's free daily limit has been reached. Try again "
                   f"later. ({exc})",
        ) from exc  # fmt: skip
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"The model call failed: {exc}") from exc
