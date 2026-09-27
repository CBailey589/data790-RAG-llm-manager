import hashlib
import json
from datetime import datetime
from typing import Any


class PromptCache:
	'''
	Simple in-memory cache for LLM responses.
	'''

	def __init__(self, max_size: int = 100):
		self.cache: dict[str, Any] = {}
		self.max_size = max_size
		self.hits = 0
		self.misses = 0

	def _hash_request(self, request: Any, model: str, temperature: float, call_type: str) -> str:
		'''
		Create a hash key for a request.
		'''
		key_data = json.dumps({
			"request": request,
			"model": model,
			"temperature": temperature,
			"call_type": call_type
		}, sort_keys=True)
		return hashlib.md5(key_data.encode()).hexdigest()

	def update_cache_max_size(self, max_size: int):
		'''
		Updates the maximum number of LLM API responses that can be held in response cache.
		'''
		if max_size <= 0:
			raise ValueError("CACHE SIZE MUST BE GREATER THAN ZERO.")

		self.max_size = max_size

		# Remove excess responses if necessary
		while len(self.cache) > self.max_size:
			oldest_key = next(iter(self.cache))
			del self.cache[oldest_key]

	def get_cached_response(self, request: Any, model: str, temperature: float, call_type: str) -> Any:
		'''
		Get cached response if available. Only available for deterministic calls (temperature = 0)
		'''
		if temperature > 0:
			self.misses += 1
			return None

		# Check for exact match:
		key = self._hash_request(request=request, model=model, temperature=temperature, call_type=call_type)
		if key in self.cache:
			self.hits += 1
			return self.cache[key]['response']

		# return None if no exact match:
		self.misses += 1
		return None

	def set_cached_response(self, request: Any, model: str, temperature: float, call_type: str, response: Any):
		'''
		Add an LLM API response to the cache. Only available for deterministic calls (temperature = 0)
		'''
		if temperature > 0:
			return

		key = self._hash_request(request=request, model=model, temperature=temperature, call_type=call_type)

		# remove oldest response if cache is full
		if len(self.cache) >= self.max_size:
			oldest_key = next(iter(self.cache))
			del self.cache[oldest_key]

		self.cache[key] = {
			"response": response,
			"timestamp": datetime.now().isoformat(),
			"model": model,
			"call_type": call_type
		}

	def get_cache_stats(self):
		'''
		Get cached response statistics.
		'''

		total_calls = self.hits + self.misses
		return {
			"size": len(self.cache),
			"hits": self.hits,
			"misses": self.misses,
			"hit_rate": self.hits / total_calls if total_calls > 0 else 0
		}
