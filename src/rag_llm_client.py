import os
from dataclasses import dataclass
from typing import Any

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .llm_prompt_validator import LLMPromptValidator


@dataclass
class RAGLLMResponse:
    success: bool
    response: Any

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

	def update_llm_model(self, model, **kwargs):
		'''
		Updates the configured LLM model for API calls.
		'''
		self.client = ChatOpenAI(
			model=model,
			base_url=self.llm_base_url,
			api_key=self.llm_api_key,
			temperature=0,
			**kwargs
		)

	def call(self, query: str, k: int) -> RAGLLMResponse:
		'''
		Performs a RAG assisted call to the LLM endpoint configured in .env using the currently configured model.
		INPUTS:
		- query: The question to ask the model

		RETURNS RAGLLMResponse class:
		- Success bool
		- RAG assisted LLM response object / returned error if unsuccessful
		'''

		try:
			valid, reason = self.prompt_validator.validate_prompt(query)
			if valid == False:
				return RAGLLMResponse(success=False, response=reason)

			qa_chain = RetrievalQA.from_chain_type(
				llm=self.client,
				chain_type="stuff",
				retriever=self.chroma.as_retriever(search_kwargs={"k": k}),
				return_source_documents=True
			)

			response = qa_chain.invoke({"query": query})
			return RAGLLMResponse(success=True, response=response)

		except Exception as e:
			return RAGLLMResponse(success=False, response=e)