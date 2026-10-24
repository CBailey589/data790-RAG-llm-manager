import os
import time
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
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
	Makes RAG assisted API calls to the configured LLM.
	'''
	def __init__(self, embedding_model: str, model: str):
		'''
		Creates an instance of the RAGLLMClient class.
		'''
		self.persistent_vectors_dir = "./chroma_db_nintendo_power"
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
			collection_name="nintendo_power_all_spreads",
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

	def _evaluate_relevant_documents(self, query: str, retrieved_docs: str ) -> tuple[float, int, int]:
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

	def call(
			self,
			query: str,
			k: int,
			verbose: bool = False,
			doc_relevance_check: bool = True,
			system_prompt: str = "You are a helpful assistant.") -> RAGLLMResponse:
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
			relevant_docs_with_scores = self.chroma.similarity_search_with_score(query, k=k)
			relevant_docs = [doc for doc, score in relevant_docs_with_scores]
			retrieved_docs_str = '\n'.join(doc.page_content for doc in relevant_docs)

			doc_relevance_score = 0.0
			evaluator_prompt_tokens = 0
			evaluator_completion_tokens = 0

			if doc_relevance_check:
				doc_relevance_score, evaluator_prompt_tokens, evaluator_completion_tokens = self._evaluate_relevant_documents(query=query, retrieved_docs=retrieved_docs_str)

			if doc_relevance_check and doc_relevance_score < self.context_relevance_threshold:
				return RAGLLMResponse(success=False, response="Unable to retrieve relevant context for API call.", latency=0, context_relevance=doc_relevance_score)
			else:
				context = "\n\n".join(
					doc.page_content for doc in relevant_docs
				)

				prompt = ChatPromptTemplate.from_messages([
					(
						"system",
						"""{system_prompt}

						CONTEXT:
						{context}"""
					),
					("human", "{question}")
				])

				messages = prompt.format_messages(
					system_prompt=system_prompt,
					context=context,
					question=query
				)

				with get_openai_callback() as cb:
					llm_response = self.client.invoke(messages)

					response = {
						"result": llm_response.content,
						"source_documents": relevant_docs,
						"usage": {
							"prompt_tokens": cb.prompt_tokens + evaluator_prompt_tokens,
							"completion_tokens": cb.completion_tokens + evaluator_completion_tokens,
							"total_tokens": cb.total_tokens
								+ evaluator_prompt_tokens
								+ evaluator_completion_tokens
						}
					}

				call_end_time = time.time()
				call_latency = (call_end_time - call_start_time) * 1000

				if verbose:
					print("\n" + "=" * 60)
					print("RAG RESPONSE")
					print("=" * 60)
					print(response["result"])

					if doc_relevance_check:
						print("\n" + "=" * 60)
						print(f"CONTEXT RELEVANCE: {doc_relevance_score:.2f}")

					print("=" * 60)
					print("\nSUPPORTING SOURCES:")
					for doc, score in relevant_docs_with_scores:
						filename = doc.metadata.get("filename", "Unknown file")
						issue = doc.metadata.get("issue_number", "Unknown issue")
						start_page = doc.metadata.get("start_page", "?")
						end_page = doc.metadata.get("end_page", "?")
						spread_id = doc.metadata.get("id", "Unknown spread id")
						preview = " ".join(doc.page_content.split())[:100]

						print(
							f"{spread_id} — Issue {issue} — "
							f"Pages {start_page}-{end_page} — Score: {score:.4f}"
						)
						print(f"    File: {filename}")
						print(f"    \"{preview}...\"")
						print()

					print("=" * 60 + "\n")

				return RAGLLMResponse(success=True, response=response, latency=call_latency, context_relevance=doc_relevance_score)

		except Exception as e:
			call_end_time = time.time()
			call_latency = (call_end_time - call_start_time) * 1000
			return RAGLLMResponse(success=False, response=e, latency=call_latency)