import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_llm_client import RAGLLMClient

load_dotenv()
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
def chat(request: ChatRequest):
	try:
		client = get_rag_client()

		result = client.call(
			query=request.message,
			k=3,
			self_rag=True,
			security_enabled=True,
		)

		if not result.success:
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
	except Exception as e:
		import traceback
		traceback.print_exc()
		raise HTTPException(
			status_code=500,
			detail=str(e),
		) from e
	# except Exception:
	#     raise HTTPException(
	#         status_code=500,
	#         detail="Unable to process RAG request.",
	#     )

