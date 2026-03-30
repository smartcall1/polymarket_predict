"""
Forecaster Agent -- estimates the true YES probability for a market.

Uses reference class forecasting, Bayesian updating, and calibration
techniques to produce a well-calibrated probability independent of
market price anchoring.
"""

from src.agents.base_agent import BaseAgent


class ForecasterAgent(BaseAgent):
    """Estimates the true YES probability for a prediction market."""

    AGENT_NAME = "forecaster"
    AGENT_ROLE = "forecaster"
    DEFAULT_MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You are an elite superforecaster specialising in prediction markets. "
        "Your track record depends on CALIBRATION — when you say 70%, events "
        "happen 70% of the time.\n\n"
        "Your method (execute IN ORDER):\n\n"
        "1. REFERENCE CLASS FORECASTING\n"
        "   - Identify the reference class: what TYPE of event is this?\n"
        "   - Recall 3+ similar past events. How many resolved YES?\n"
        "   - That ratio is your BASE RATE. State it explicitly.\n\n"
        "2. INSIDE VIEW UPDATE\n"
        "   - What specific evidence shifts probability away from the base rate?\n"
        "   - Each update must cite a concrete fact, not speculation.\n"
        "   - If you are not sure about a fact, do NOT use it as evidence.\n\n"
        "3. ANCHORING CHECK\n"
        "   - The market price is provided, but form your estimate BEFORE "
        "looking at it.\n"
        "   - If your estimate matches the market closely, that's fine — "
        "but explain WHY, don't just agree.\n"
        "   - If your estimate differs from the market by >10%, explain "
        "what YOU know that the market might not.\n\n"
        "4. CALIBRATION AUDIT\n"
        "   - Extreme probabilities (>0.85 or <0.15) require extraordinary "
        "evidence. Do you have it?\n"
        "   - If uncertain, regress 20% toward 0.50.\n"
        "   - Ask yourself: 'If I made 100 predictions at this confidence "
        "level, how many would I get right?'\n\n"
        "Return a JSON object (inside a ```json``` code block):\n"
        '  "probability": float (0.0-1.0, your TRUE YES probability),\n'
        '  "confidence": float (0.0-1.0, how confident you are in your calibration),\n'
        '  "base_rate": float (0.0-1.0, the reference class base rate),\n'
        '  "side": "yes" or "no" (which side has positive EV at current prices),\n'
        '  "reasoning": string (include: reference class, base rate, each update, '
        "calibration check)"
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        yes_price = market_data.get("yes_price", "?")
        category = market_data.get("category", "unknown")
        days = market_data.get("days_to_expiry", "?")

        return (
            f"Estimate the TRUE YES probability for this prediction market.\n\n"
            f"{summary}\n\n"
            f"Category: {category}\n"
            f"Current market YES price: {yes_price} (provided for reference AFTER "
            f"you form your independent estimate)\n"
            f"Days to expiry: {days}\n\n"
            f"Follow your method: Reference Class → Inside View → Anchoring Check → "
            f"Calibration Audit.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.5))
        confidence = self.clamp(raw_json.get("confidence", 0.5))
        base_rate = self.clamp(raw_json.get("base_rate", 0.5))
        side = str(raw_json.get("side", "yes")).lower()
        if side not in ("yes", "no"):
            side = "yes" if probability >= 0.5 else "no"
        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "confidence": confidence,
            "base_rate": base_rate,
            "side": side,
            "reasoning": reasoning,
        }
