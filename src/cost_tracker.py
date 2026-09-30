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
	call_type: str = "llm"


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
		error: str = None,
		call_type: str = "llm"
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
				response=response,
				call_type=call_type
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
				response=response,
				call_type=call_type,
			)

		self.records.append(record)
		return record

	def get_summary_of_tracked_llm_calls(self) -> dict:
		'''
		Get summary statistics.
		'''
		if not self.records:
			return {"error": "No records"}

		df = pd.DataFrame([
			{
				'timestamp': r.timestamp,
				'model': r.model,
				'prompt_tokens': r.prompt_tokens,
				'completion_tokens': r.completion_tokens,
				'total_tokens': r.total_tokens,
				'cost_usd': r.cost_usd,
				'latency_ms': r.latency_ms,
				'success': r.success,
				'cache_hit': r.cache_hit
			}
			for r in self.records
		])

		api_calls = df[df["cache_hit"] == False]

		return {
			"total_calls": len(self.records),
			"successful_calls": df["success"].sum(),
			"total_tokens": api_calls["total_tokens"].sum(),
			"total_cost_usd": api_calls["cost_usd"].sum(),
			"cost_per_call": api_calls["cost_usd"].mean(),
			"cost_by_model": api_calls.groupby("model")["cost_usd"].sum().to_dict(),
			"avg_latency_ms": api_calls["latency_ms"].mean(),
			"latency_by_model": api_calls.groupby("model")["latency_ms"].mean().to_dict(),
			"cache_hit_rate": df["cache_hit"].mean() if "cache_hit" in df else 0
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
		for model, cost in s['cost_by_model'].items():
			print(f"  {model}: ${cost:.6f}")

	def project_rag_costs(self):
		'''
		Project RAG API costs at different daily request volumes.
		Uses only successful, uncached RAG calls for the currently configured model.
		'''
		df = self.get_dataframe_of_tracked_llm_calls()

		rag_calls = df[
			(df["model"] == self.model) &
			(df["call_type"] == "rag") &
			(df["success"]) &
			(~df["cache_hit"])
		]

		if rag_calls.empty:
			print(f"No uncached RAG calls recorded for model: {self.model}")
			return

		average_cost = rag_calls["cost_usd"].mean()

		print("\n" + "=" * 70)
		print("PROJECTED RAG COSTS")
		print("=" * 70)
		print(f"Model: {self.model}")
		print(f"Average RAG cost per call: ${average_cost:.6f}")
		print(f"Based on: {len(rag_calls)} recorded RAG calls")

		print("\nDaily Calls     Per Day     Per Week     Per Month     Per Year")
		print("-" * 70)

		for daily_calls in [1_000, 10_000, 100_000]:
			daily_cost = daily_calls * average_cost
			weekly_cost = daily_cost * 7
			monthly_cost = daily_cost * 30
			yearly_cost = daily_cost * 365

			print(
				f"{daily_calls:>11,}"
				f"   ${daily_cost:>8.2f}"
				f"   ${weekly_cost:>9.2f}"
				f"   ${monthly_cost:>10.2f}"
				f"   ${yearly_cost:>10.2f}"
			)

		print("=" * 70)