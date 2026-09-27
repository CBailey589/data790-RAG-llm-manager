import os
import time
from dataclasses import dataclass
from typing import Any

from langchain.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .llm_prompt_validator import LLMPromptValidator


@dataclass
class RAGLLMResponse:
	success: bool
	response: Any
	latency: float = 0
	context_relevance: float = 0

class RAGLLMClient:
	'''
	MAKES RAG assisted API calls to the configured LLM.
	'''
	def __init__(self, embedding_model: str, model: str):
		'''
		Creates an instance of the RAGLLMClient class.
		'''
		self.persistent_vectors_dir = "./chromadb"
		self.llm_base_url = os.getenv("LLM_BASE_URL")
		self.llm_api_key = os.getenv("LLM_API_KEY")
		self.client = ChatOpenAI(
			model=model,
			base_url=self.llm_base_url,
			api_key=self.llm_api_key,
			temperature=0,
		)
		self.embeddings = OpenAIEmbeddings(
			model=embedding_model,
			base_url=self.llm_base_url,
			api_key=self.llm_api_key,
		)
		self.chroma = Chroma(
			persist_directory=self.persistent_vectors_dir,
			embedding_function=self.embeddings,
		)
		self.prompt_validator = LLMPromptValidator()
		self.context_relevance_evaluator = ChatOpenAI(
			model="gpt-4.1-mini",
			base_url=self.llm_base_url,
			api_key=self.llm_api_key,
			temperature=0,
		)
		self.context_relevance_threshold = 0.7

	def update_llm_model(self, model, temperature=0, **kwargs):
		'''
		Updates the configured LLM model for API calls.
		'''
		self.client = ChatOpenAI(
			model=model,
			base_url=self.llm_base_url,
			api_key=self.llm_api_key,
			temperature=temperature,
			**kwargs
		)

	def _evaluate_relevant_documents(self, query: str, retrieved_docs: str ) -> tuple(float, int, int):
		try:
			evaluator_prompt = f'''
			Evaluate how relevant the following retrieved documents are to the users question.

			USER'S QUESTION:
			{query}

			RETRIEVED DOCUMENTS:
			{retrieved_docs}

			RETURN ONLY A NUMBER BETWEEN 0.0 AND 1.0. IT IS VERY IMPORTANT THAT YOU FOLLOW THIS.
			0.0 means the retrieved documents are not at all relevant.
			1.0 means the retrieved documents contain the information necessary to answer the question.
			'''

			# Counts towards daily budget:
			with get_openai_callback() as cb:
				evaluator_response = self.context_relevance_evaluator.invoke(
					evaluator_prompt
				)

				prompt_tokens = cb.prompt_tokens
				completion_tokens = cb.completion_tokens

			context_relevance = float(evaluator_response.content.strip())
			return context_relevance, prompt_tokens, completion_tokens

		except Exception as e:
			return 0.0, 0, 0

	def call(self, query: str, k: int) -> RAGLLMResponse:
		'''
		Performs a RAG assisted call to the LLM endpoint configured in .env using the currently configured model.
		INPUTS:
		- query: The question to ask the model

		RETURNS RAGLLMResponse class:
		- Success bool
		- RAG assisted LLM response object / returned error if unsuccessful
		'''

		valid, reason = self.prompt_validator.validate_prompt(query)
		if valid == False:
			return RAGLLMResponse(success=False, response=reason)

		call_start_time = time.time()

		try:
			# Get documents to evaluate their relevance:
			document_retriever = self.chroma.as_retriever(search_kwargs={"k": k})
			relevant_docs = document_retriever.invoke(query)
			retrieved_docs_str = '\n'.join(doc.page_content for doc in relevant_docs)

			doc_relevance_score, evaluator_prompt_tokens, evaluator_completion_tokens = self._evaluate_relevant_documents(query=query, retrieved_docs=retrieved_docs_str)

			if doc_relevance_score < self.context_relevance_threshold:
				return RAGLLMResponse(success=False, response="Unable to retrieve relevant context for API call.", latency=0, context_relevance=doc_relevance_score)
			else:
				qa_chain = RetrievalQA.from_chain_type(
					llm=self.client,
					chain_type="stuff",
					retriever=self.chroma.as_retriever(search_kwargs={"k": k}),
					return_source_documents=True
				)

				with get_openai_callback() as cb:
					response = qa_chain.invoke({"query": query})
					response["usage"] = {
						"prompt_tokens": cb.prompt_tokens + evaluator_prompt_tokens,
						"completion_tokens": cb.completion_tokens + evaluator_completion_tokens,
						"total_tokens": cb.total_tokens + evaluator_prompt_tokens + evaluator_completion_tokens
					}

				call_end_time = time.time()
				call_latency = (call_end_time - call_start_time) * 1000

				return RAGLLMResponse(success=True, response=response, latency=call_latency)

		except Exception as e:
			call_end_time = time.time()
			call_latency = (call_end_time - call_start_time) * 1000
			return RAGLLMResponse(success=False, response=e, latency=call_latency)