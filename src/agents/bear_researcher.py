"""
Bear Researcher Agent -- argues for NO with a market-efficiency lens.

Operates independently in ensemble mode. Combines skepticism with
the assumption that market prices already reflect collective wisdom.
"""

from src.agents.base_agent import BaseAgent


class BearResearcher(BaseAgent):
    """Researches and presents the bearish (NO) case for a market."""

    AGENT_NAME = "bear_researcher"
    AGENT_ROLE = "bear_researcher"
    DEFAULT_MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You are a sceptical risk analyst and market-efficiency advocate "
        "in a prediction market fund. Your dual role:\n"
        "1. Make the STRONGEST case that this event will NOT happen (NO).\n"
        "2. Argue that the current market price is ALREADY CORRECT or even "
        "too generous to YES.\n\n"
        "You are one voice in a multi-agent ensemble. A bull analyst will "
        "argue the opposite. Your job is to be the toughest critic.\n\n"
        "METHOD:\n"
        "1. MARKET EFFICIENCY ARGUMENT\n"
        "   - The market price reflects the bets of many participants.\n"
        "   - Explain WHY the current price might already be correct.\n"
        "   - What information is ALREADY priced in?\n\n"
        "2. DEVIL'S ADVOCATE\n"
        "   - Think of the 3 strongest arguments for YES.\n"
        "   - Now refute EACH ONE with specific counter-evidence.\n\n"
        "3. STRUCTURAL BARRIERS\n"
        "   - What structural, logistical, or historical factors make YES "
        "unlikely?\n"
        "   - Cite specific precedents where similar events failed.\n\n"
        "4. PROBABILITY CEILING\n"
        "   - Even if every bull argument is correct, what is the MAXIMUM "
        "reasonable YES probability?\n"
        "   - What structural barrier creates this ceiling?\n\n"
        "RULES:\n"
        "- Use ONLY facts you are confident about. No speculation.\n"
        "- Each argument must include specific evidence.\n"
        "- A strong bear case with honest confidence > a fabricated one.\n\n"
        "Return a JSON object (inside a ```json``` code block):\n"
        '  "probability": float (0.0-1.0, your YES probability — typically lower),\n'
        '  "probability_ceiling": float (0.0-1.0, max reasonable YES probability),\n'
        '  "confidence": float (0.0-1.0, confidence in your analysis),\n'
        '  "key_arguments": list of strings (3-5 arguments for NO),\n'
        '  "risk_factors": list of strings (risks for YES holders),\n'
        '  "reasoning": string (detailed bear thesis including market-efficiency argument)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        yes_price = market_data.get("yes_price", "?")
        days = market_data.get("days_to_expiry", "?")
        category = market_data.get("category", "unknown")

        return (
            f"Make the STRONGEST case that this market resolves NO, and argue "
            f"that the current market price is already fair or too generous.\n\n"
            f"{summary}\n\n"
            f"Category: {category}\n"
            f"Current YES price: {yes_price} (this price reflects many bettors' "
            f"collective judgement — explain why they might be right)\n"
            f"Days to expiry: {days}\n\n"
            f"Start by explaining why the market price could be correct, "
            f"then build your bear case.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.4))
        probability_ceiling = self.clamp(raw_json.get("probability_ceiling", 0.7))
        confidence = self.clamp(raw_json.get("confidence", 0.5))

        key_arguments = raw_json.get("key_arguments", [])
        if not isinstance(key_arguments, list):
            key_arguments = [str(key_arguments)]
        key_arguments = [str(a) for a in key_arguments][:10]

        risk_factors = raw_json.get("risk_factors", [])
        if not isinstance(risk_factors, list):
            risk_factors = [str(risk_factors)]
        risk_factors = [str(r) for r in risk_factors][:10]

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "probability_ceiling": probability_ceiling,
            "confidence": confidence,
            "key_arguments": key_arguments,
            "risk_factors": risk_factors,
            "reasoning": reasoning,
        }
