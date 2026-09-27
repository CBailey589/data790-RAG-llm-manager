class BudgetManager:
	'''
	Budget Management for LLM API Calls.
	'''

	def __init__(self, daily_budget: float, enable_budget_hard_cap: bool):
		self.daily_budget = daily_budget
		self.budget_hard_cap_enabled = enable_budget_hard_cap
		self.daily_spend = 0
		self.previous_percentage = 0.0
		self.current_percentage = 0.0

	def update_daily_budget(self, daily_budget: float):
		'''
		Allows a user to update their daily budget for LLM API calls.
		'''
		if daily_budget <= 0:
			raise ValueError("[$$$] DAILY BUDGET MUST BE GREATER THAN ZERO.")

		prev_budget = self.daily_budget
		self.daily_budget = daily_budget
		print(f"[$$$] DAILY BUDGET UPDATED FROM ${round(prev_budget,4)} -> ${round(daily_budget,4)}...")
		self.check_daily_budget_remaining()

	def enable_budget_hard_cap(self):
		'''
		Turns on budget hard cap that prevents LLM API calls if daily budget is exceeded.
		'''
		self.budget_hard_cap_enabled = True
		print("[$$$] Budget hard cap enabled.")

	def disable_budget_hard_cap(self):
		'''
		Turns off budget hard cap that prevents LLM API calls if daily budget is exceeded.
		'''
		self.budget_hard_cap_enabled = False
		print("[$$$] Budget hard cap disabled.")

	def check_daily_budget_remaining(self) -> bool:
		'''
		Checks a user's LLM API call expenditure against their configured daily budget.
		'''
		if self.daily_spend >= self.daily_budget:
			print(f"[$$$] DAILY BUDGET EXCEEDED! Budget: ${round(self.daily_budget,4)}, Utilized ${round(self.daily_spend,4)}")
		else:
			print(f"[$$$] ${round(self.daily_budget - self.daily_spend,4)} of daily budget remaining. Budget: ${round(self.daily_budget,4)}, Utilized ${round(self.daily_spend,4)}")

		return self.allow_llm_api_call()

	def allow_llm_api_call(self) -> bool:
		'''
		Returns a bool indicating if the user has money remaining within their daily budget to make an LLM API call.
		'''
		if self.budget_hard_cap_enabled and self.daily_budget <= self.daily_spend:
			print("[$$$] API CALL BLOCKED. To continue with LLM API calls disable the daily budget hard cap using the disable_budget_hard_cap() method.")
			return False
		else:
			return True

	def record_cost(self, cost: float):
		'''
		Updates a user's daily expenditure with a new cost.
		'''
		self.daily_spend = self.daily_spend + cost
		pct_budget_utilized = self.daily_spend / self.daily_budget
		self.previous_percentage = self.current_percentage
		self.current_percentage = pct_budget_utilized

		# Alert user if 50%, 80%, or 100% of daily budget has been used
		if self.previous_percentage < 1.0 and self.current_percentage >= 1.0:
			print("!"*20)
			print(f"[$$$] 100% of daily budget utilized. [Budget: ${round(self.daily_budget,4)}, Utilized: ${round(self.daily_spend,4)}]")
			print("!"*20)
		elif self.previous_percentage < 0.8 and self.current_percentage >= 0.8:
			print("!"*20)
			print(f"[$$$] 80% of daily budget utilized. [Budget: ${round(self.daily_budget,4)}, Utilized: ${round(self.daily_spend,4)}]")
			print("!"*20)
		elif self.previous_percentage < 0.5 and self.current_percentage >= 0.5:
			print("!"*20)
			print(f"[$$$] 50% of daily budget utilized. [Budget: ${round(self.daily_budget,4)}, Utilized: ${round(self.daily_spend,4)}]")
			print("!"*20)
