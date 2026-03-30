"""
Trader Agent -- final decision gate after ensemble voting.

Runs on a stronger model (gemini-3.1-pro-preview) and receives all
ensemble agents' outputs. Acts as a conservative final gatekeeper:
the ensemble already suggests BUY, so the Trader's job is to CONFIRM
or REJECT with independent judgement.
"""

from src.agents.base_agent import BaseAgent


class TraderAgent(BaseAgent):
    """
    Synthesises all agents' outputs into a final BUY/SELL/SKIP decision.
    Acts as the last line of defence — conservative by design.
    """

    AGENT_NAME = "trader"
    AGENT_ROLE = "trader"
    DEFAULT_MODEL = "gemini-3.1-pro-preview"

    SYSTEM_PROMPT = (
        "You are the HEAD TRADER at an AI prediction-market fund. You are "
        "the FINAL GATEKEEPER — your team of analysts has already voted to "
        "BUY, and you must independently decide whether to CONFIRM or REJECT.\n\n"
        "IMPORTANT CONTEXT:\n"
        "- You are receiving BIASED input: the ensemble already filtered for "
        "trades with edge. Be aware of this selection bias.\n"
        "- Your job is to find reasons the ensemble might be WRONG.\n"
        "- When in doubt, SKIP. Preserving capital is always the priority.\n\n"
        "DECISION FRAMEWORK:\n"
        "1. CONSENSUS CHECK\n"
        "   - Review the ensemble probability and disagreement.\n"
        "   - High disagreement (>0.20 std dev) = low conviction → lean SKIP.\n"
        "   - Check: do the bull floor and bear ceiling overlap? If not, "
        "the agents fundamentally disagree.\n\n"
        "2. EDGE VERIFICATION\n"
        "   - Recalculate: edge = ensemble_probability - market_price.\n"
        "   - Is the edge real, or based on stale/wrong information?\n"
        "   - Would this edge survive if one key assumption is wrong?\n\n"
        "3. RISK CHECK\n"
        "   - Does the risk manager approve (should_trade=true)?\n"
        "   - Is risk_score <= 7?\n"
        "   - Is the volume sufficient for our position size?\n\n"
        "4. FINAL CALL\n"
        "   - BUY only if: edge >= 5%, confidence >= 60%, risk acceptable, "
        "AND you cannot find a strong reason to reject.\n"
        "   - SKIP if: any doubt remains. You will see hundreds more "
        "opportunities.\n\n"
        "Return your decision as a JSON object (inside a ```json``` code block):\n"
        '  "action": "BUY" | "SELL" | "SKIP",\n'
        '  "side": "YES" | "NO",\n'
        '  "limit_price": int (cents, 1-99),\n'
        '  "confidence": float (0.0-1.0),\n'
        '  "position_size_pct": float (percent of capital to risk),\n'
        '  "reasoning": string (must reference specific agent outputs and '
        "explain why you confirm or reject)"
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        # Assemble each agent's results into a structured briefing
        briefing_parts = []

        # Ensemble meta (from the ensemble runner)
        meta = context.get("ensemble_meta")
        if meta:
            briefing_parts.append(
                f"ENSEMBLE SUMMARY:\n"
                f"  Weighted probability: {meta.get('probability', '?')}\n"
                f"  Aggregate confidence: {meta.get('confidence', '?')}\n"
                f"  Disagreement (std dev): {meta.get('disagreement', '?')}\n"
                f"  Suggested side: {meta.get('suggested_side', '?')}\n"
                f"  Calculated edge: {meta.get('edge', '?')}\n"
                f"  Models used: {meta.get('num_models', '?')}"
            )

        if context.get("forecaster_result"):
            fc = context["forecaster_result"]
            briefing_parts.append(
                f"FORECASTER:\n"
                f"  YES probability: {fc.get('probability', '?')}\n"
                f"  Confidence: {fc.get('confidence', '?')}\n"
                f"  Base rate: {fc.get('base_rate', '?')}\n"
                f"  Reasoning: {fc.get('reasoning', 'N/A')[:400]}"
            )

        if context.get("news_result"):
            news = context["news_result"]
            factors = ", ".join(news.get("key_factors", [])[:5])
            briefing_parts.append(
                f"CONTEXT ANALYST:\n"
                f"  Sentiment: {news.get('sentiment', '?')}\n"
                f"  Relevance: {news.get('relevance', '?')}\n"
                f"  Impact: {news.get('impact_direction', '?')}\n"
                f"  Key factors: {factors}\n"
                f"  Reasoning: {news.get('reasoning', 'N/A')[:400]}"
            )

        if context.get("bull_result"):
            bull = context["bull_result"]
            args = "; ".join(bull.get("key_arguments", [])[:5])
            briefing_parts.append(
                f"BULL RESEARCHER:\n"
                f"  YES probability: {bull.get('probability', '?')}\n"
                f"  Probability floor: {bull.get('probability_floor', '?')}\n"
                f"  Confidence: {bull.get('confidence', '?')}\n"
                f"  Arguments: {args}\n"
                f"  Reasoning: {bull.get('reasoning', 'N/A')[:400]}"
            )

        if context.get("bear_result"):
            bear = context["bear_result"]
            args = "; ".join(bear.get("key_arguments", [])[:5])
            briefing_parts.append(
                f"BEAR RESEARCHER:\n"
                f"  YES probability: {bear.get('probability', '?')}\n"
                f"  Probability ceiling: {bear.get('probability_ceiling', '?')}\n"
                f"  Confidence: {bear.get('confidence', '?')}\n"
                f"  Arguments: {args}\n"
                f"  Reasoning: {bear.get('reasoning', 'N/A')[:400]}"
            )

        if context.get("risk_result"):
            risk = context["risk_result"]
            briefing_parts.append(
                f"RISK MANAGER:\n"
                f"  Risk score: {risk.get('risk_score', '?')}/10\n"
                f"  Recommended size: {risk.get('recommended_size_pct', '?')}%\n"
                f"  EV estimate: {risk.get('ev_estimate', '?')}\n"
                f"  Should trade: {risk.get('should_trade', '?')}\n"
                f"  Reasoning: {risk.get('reasoning', 'N/A')[:400]}"
            )

        briefing = "\n\n".join(briefing_parts) if briefing_parts else "[No agent analyses available]"

        return (
            f"Your team has voted to BUY. Review their analysis and make "
            f"the FINAL trading decision.\n\n"
            f"=== MARKET ===\n{summary}\n\n"
            f"=== TEAM ANALYSIS ===\n{briefing}\n\n"
            f"Remember: you are the last line of defence. The team is biased "
            f"toward BUY because only BUY signals reach you. "
            f"Look for reasons to REJECT.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        action = str(raw_json.get("action", "SKIP")).upper()
        if action not in ("BUY", "SELL", "SKIP"):
            action = "SKIP"

        side = str(raw_json.get("side", "YES")).upper()
        if side not in ("YES", "NO"):
            side = "YES"

        confidence = self.clamp(raw_json.get("confidence", 0.5))

        try:
            limit_price = int(raw_json.get("limit_price", 50))
            limit_price = max(1, min(99, limit_price))
        except (TypeError, ValueError):
            limit_price = 50

        position_size_pct = self.clamp(
            raw_json.get("position_size_pct", 1.0), lo=0.0, hi=100.0
        )

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "action": action,
            "side": side,
            "limit_price": limit_price,
            "confidence": confidence,
            "position_size_pct": position_size_pct,
            "reasoning": reasoning,
        }
