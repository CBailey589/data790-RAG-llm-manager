import logging
import os
import secrets
from functools import lru_cache

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.rag_llm_client import RAGLLMClient

load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment("nintendo-power-rag")
	mlflow.langchain.autolog()
app = FastAPI(title="Nintendo Power RAG API")

class ChatRequest(BaseModel):
	message: str = Field(min_length=1, max_length=10000)


@lru_cache(maxsize=1)
def get_rag_client() -> RAGLLMClient:
	return RAGLLMClient(
		embedding_model=os.environ["DEFAULT_EMBEDDINGS_MODEL"],
		model=os.environ["DEFAULT_MODEL"],
	)


@app.get("/health")
def health():
	return {"status": "running"}


@app.post("/chat")
def chat(request: ChatRequest,
	x_eval_suite: str | None = Header(default=None),
	x_eval_key: str | None = Header(default=None),
	x_eval_test_id: str | None = Header(default=None)
):
	try:
		suite = "interactive"
		test_id = None
		expected_key = os.getenv("EVALUATION_API_KEY")

		if (x_eval_suite and expected_key and x_eval_key and secrets.compare_digest(x_eval_key, expected_key)):
			suite = x_eval_suite
			test_id = x_eval_test_id

		client = get_rag_client()

		with mlflow.start_span(name="Nintendo Power RAG Request") as span:
			tags = {"test_suite": suite}
			if test_id:
				tags["test_id"] = test_id
			mlflow.update_current_trace(tags=tags)
			span.set_attribute("retrieval_k", 3)
			span.set_attribute("self_rag_enabled", True)
			span.set_attribute("security_enabled", True)

			result = client.call(
				query=request.message,
				k=3,
				self_rag=True,
				security_enabled=True,
			)

			span.set_attribute("success", result.success)
			span.set_attribute("security_blocked", result.security_blocked)
			span.set_attribute("latency_ms", result.latency)
			span.set_attribute(
				"retrieved_document_count",
				len(result.retrieved_spread_ids),
			)


		if not result.success:
			logging.error("RAG request failed: %s", result.response)

			if result.security_blocked:
				raise HTTPException(
					status_code=403,
					detail="Request rejected by security validation.",
				)

			raise HTTPException(
				status_code=500,
				detail="RAG request failed.",
			)

		return {
			"response": result.response,
			"latency_ms": result.latency,
			"retrieved_spread_ids": result.retrieved_spread_ids,
		}

	except HTTPException:
		raise
	except Exception:
		logging.exception("Unexpected RAG API error")
		raise HTTPException(
			status_code=500,
			detail="Unable to process RAG request.",
		) from None

