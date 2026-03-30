"""
Bull Researcher Agent -- makes the strongest evidence-based case for YES.

Operates independently in ensemble mode (no access to other agents' outputs).
Focuses on verifiable facts and avoids speculation.
"""

from src.agents.base_agent import BaseAgent


class BullResearcher(BaseAgent):
    """Researches and presents the bullish (YES) case for a market."""

    AGENT_NAME = "bull_researcher"
    AGENT_ROLE = "bull_researcher"
    DEFAULT_MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You are a conviction-driven research analyst in a prediction market "
        "fund. Your job is to build the STRONGEST possible case that this "
        "event WILL happen (YES outcome).\n\n"
        "You are one voice in a multi-agent ensemble. Another analyst will "
        "argue the opposite. Your job is not to be balanced — be the best "
        "advocate for YES.\n\n"
        "RULES:\n"
        "- Use ONLY facts you are confident about. No speculation or hedging.\n"
        "- Each argument must include a specific date, number, name, or "
        "verifiable claim.\n"
        "- If you cannot find strong evidence, say so — a weak bull case "
        "with low confidence is more useful than fabricated arguments.\n"
        "- Consider the TIMELINE: is there enough time before expiry for "
        "YES to happen?\n\n"
        "Structure:\n"
        "1. THESIS — One sentence: why YES will happen.\n"
        "2. KEY ARGUMENTS — 3-5 concrete, evidence-based reasons.\n"
        "3. PROBABILITY FLOOR — The minimum YES probability even if every "
        "bear argument is correct. What scenario guarantees this floor?\n"
        "4. CATALYSTS — Near-term events that could push probability higher.\n\n"
        "Return a JSON object (inside a ```json``` code block):\n"
        '  "probability": float (0.0-1.0, your YES probability estimate),\n'
        '  "probability_floor": float (0.0-1.0, minimum YES probability),\n'
        '  "confidence": float (0.0-1.0, confidence in your analysis quality),\n'
        '  "key_arguments": list of strings (3-5 evidence-based arguments),\n'
        '  "catalysts": list of strings (near-term bullish catalysts),\n'
        '  "reasoning": string (detailed bull thesis)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        days = market_data.get("days_to_expiry", "?")
        category = market_data.get("category", "unknown")

        return (
            f"Build the STRONGEST evidence-based case that this market "
            f"resolves YES.\n\n"
            f"{summary}\n\n"
            f"Category: {category}\n"
            f"Days to expiry: {days}\n\n"
            f"Remember: only use facts you are confident about. "
            f"Weak evidence with honest low confidence > fabricated strong evidence.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.6))
        probability_floor = self.clamp(raw_json.get("probability_floor", 0.3))
        confidence = self.clamp(raw_json.get("confidence", 0.5))

        key_arguments = raw_json.get("key_arguments", [])
        if not isinstance(key_arguments, list):
            key_arguments = [str(key_arguments)]
        key_arguments = [str(a) for a in key_arguments][:10]

        catalysts = raw_json.get("catalysts", [])
        if not isinstance(catalysts, list):
            catalysts = [str(catalysts)]
        catalysts = [str(c) for c in catalysts][:10]

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "probability_floor": probability_floor,
            "confidence": confidence,
            "key_arguments": key_arguments,
            "catalysts": catalysts,
            "reasoning": reasoning,
        }
