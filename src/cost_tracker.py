from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .pricing_db import PRICING_DB


@dataclass
class APICallRecord:
	"""Record of a single API call."""
	timestamp: str
	model: str
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int
	cost_usd: float
	latency_ms: float
	success: bool
	response: Any
	error: Optional[str] = None
	cache_hit: bool = False


class CostTracker:
	'''
	Comprehensive cost tracking for LLM API calls.
	'''

	def __init__(self, model: str):
		self.pricing = PRICING_DB
		self.records: list[APICallRecord] = []
		self.model = model

	def update_model(self, model: str):
		if model in self.pricing:
			self.model = model
		else:
			raise ValueError("Provided model not in Pricing Database. Utilize insert_model_into_pricing_db() prior to setting a new model.")

	def insert_model_into_pricing_db(self, model_name: str, input_cost: int, output_cost: int, provider: str):
		'''
		Inserts a new record into the pricing database for cost tracking.
		'''
		self.pricing[model_name] = {
			"input":input_cost,
			"output":output_cost,
			"provider":provider
		}

	def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
		'''
		Calculate cost for a given usage.
		'''

		prices = self.pricing[self.model]
		input_cost = (prompt_tokens / 1000) * prices["input"]
		output_cost = (completion_tokens / 1000) * prices["output"]
		return input_cost + output_cost

	def record_call(
		self,
		response,
		latency_ms: float,
		prompt_tokens: int,
		completion_tokens: int,
		total_tokens: int,
		cache_hit: bool = False,
		error: str = None
	):
		'''
		Record an API call.
		'''
		if error:
			record = APICallRecord(
				timestamp=datetime.now().isoformat(),
				model=self.model,
				prompt_tokens=0,
				completion_tokens=0,
				total_tokens=0,
				cost_usd=0,
				latency_ms=latency_ms,
				success=False,
				error=error,
				cache_hit=cache_hit,
				response=response
			)
		else:
			# Zero cost of cached_responses:
			if cache_hit:
				cost = 0
			else:
				cost = self.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

			record = APICallRecord(
				timestamp=datetime.now().isoformat(),
				model=self.model,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=total_tokens,
				cost_usd=cost,
				latency_ms=latency_ms,
				success=True,
				cache_hit=cache_hit,
				response=response
			)

		self.records.append(record)
		return record

	def get_summary_of_tracked_llm_calls(self) -> dict:
		'''
		Get summary statistics.
		'''
		if not self.records:
			return {"error": "No records"}

		df = pd.DataFrame([asdict(r) for r in self.records])

		return {
			"total_calls": len(self.records),
			"successful_calls": df["success"].sum(),
			"total_tokens": df["total_tokens"].sum(),
			"total_cost_usd": df["cost_usd"].sum(),
			"avg_latency_ms": df["latency_ms"].mean(),
			"cache_hit_rate": df["cache_hit"].mean() if "cache_hit" in df else 0,
			"cost_per_call": df["cost_usd"].mean(),
			"by_model": df.groupby("model")["cost_usd"].sum().to_dict()
		}

	def get_dataframe_of_tracked_llm_calls(self) -> pd.DataFrame:
		'''
		Convert records to DataFrame.
		'''
		return pd.DataFrame([asdict(r) for r in self.records])

	def print_summary_of_tracked_llm_calls(self):
		'''
		Print formatted summary.
		'''
		if not self.records:
			print("No API calls have been recorded.")
			return

		s = self.get_summary_of_tracked_llm_calls()
		print("\n" + "="*50)
		print("COST TRACKING SUMMARY")
		print("="*50)
		print(f"Total API calls: {s['total_calls']}")
		print(f"Successful calls: {s['successful_calls']}")
		print(f"Total tokens: {s['total_tokens']:,}")
		print(f"Total cost: ${s['total_cost_usd']:.6f}")
		print(f"Average latency: {s['avg_latency_ms']:.0f}ms")
		print(f"Cache hit rate: {s['cache_hit_rate']:.1%}")
		print(f"Average cost per call: ${s['cost_per_call']:.6f}")
		print("\nCost by model:")
		for model, cost in s['by_model'].items():
			print(f"  {model}: ${cost:.6f}")