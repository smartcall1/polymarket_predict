"""
Risk Manager Agent -- evaluates risk/reward with prediction-market-specific
risk factors including liquidity, slippage, and time decay.

Operates independently in ensemble mode and also provides its own
probability estimate for the weighted vote.
"""

from src.agents.base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """Evaluates risk/reward profile and recommends position sizing."""

    AGENT_NAME = "risk_manager"
    AGENT_ROLE = "risk_manager"
    DEFAULT_MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You are a quantitative risk manager for a prediction-market trading "
        "desk on Predict.fun (BNB Chain). Your job is to independently assess "
        "whether a trade has acceptable risk/reward.\n\n"
        "PREDICTION MARKET EV FORMULA:\n"
        "  BUY YES EV = P(YES) - yes_price\n"
        "  BUY NO  EV = P(NO)  - no_price = (1 - P(YES)) - (1 - yes_price)\n"
        "  Trade only if |EV| >= 0.05 (5% minimum edge).\n\n"
        "RISK ASSESSMENT (rate each 1-10, then sum for total risk_score):\n"
        "1. LIQUIDITY RISK — Volume < $5,000 → high slippage risk (+2). "
        "Volume < $1,000 → untradeable (+5).\n"
        "2. TIME RISK — Days to expiry < 7 → time value decay (+2). "
        "Days < 1 → extreme risk (+3).\n"
        "3. INFORMATION RISK — How reliable is the available information? "
        "Crypto/sports have fast-moving info (+1-2).\n"
        "4. PLATFORM RISK — Predict.fun on BNB Chain: consider settlement "
        "delays, lower liquidity vs Polymarket (+1).\n\n"
        "POSITION SIZING:\n"
        "- Use fractional Kelly: size_pct = (edge / odds) * 0.25\n"
        "- Never exceed 5% of portfolio on one trade.\n"
        "- Higher risk_score → proportionally smaller size.\n\n"
        "You must also provide your OWN independent probability estimate "
        "to contribute to the ensemble vote.\n\n"
        "Return a JSON object (inside a ```json``` code block):\n"
        '  "probability": float (0.0-1.0, your independent YES probability),\n'
        '  "risk_score": float (1.0-10.0, total risk),\n'
        '  "recommended_size_pct": float (0.0-5.0, percent of capital),\n'
        '  "ev_estimate": float (expected value as decimal, e.g. 0.08 = 8%),\n'
        '  "max_loss_pct": float (worst case loss as percent of position),\n'
        '  "edge_durability_hours": float (how long the edge lasts),\n'
        '  "should_trade": boolean (true if risk/reward is acceptable),\n'
        '  "reasoning": string (detailed risk analysis with each risk component)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        volume = market_data.get("volume", 0)
        days = market_data.get("days_to_expiry", "?")
        yes_price = market_data.get("yes_price", "?")
        no_price = market_data.get("no_price", "?")
        category = market_data.get("category", "unknown")

        # Portfolio context if available
        portfolio_section = ""
        if context.get("portfolio"):
            pf = context["portfolio"]
            portfolio_section = (
                f"\n\nPortfolio: cash=${pf.get('cash', 0):,.2f}, "
                f"max_position_pct={pf.get('max_position_pct', 5)}%, "
                f"existing_positions={pf.get('existing_positions', 0)}"
            )

        return (
            f"Evaluate risk/reward for this Predict.fun trade.\n\n"
            f"{summary}\n\n"
            f"Category: {category}\n"
            f"YES price: {yes_price} | NO price: {no_price}\n"
            f"Volume (USD): ${volume:,.0f}\n"
            f"Days to expiry: {days}\n"
            f"{portfolio_section}\n\n"
            f"Calculate EV, assess each risk component, recommend sizing.\n"
            f"Also provide your independent probability estimate.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.5))
        risk_score = self.clamp(raw_json.get("risk_score", 5.0), lo=1.0, hi=10.0)
        recommended_size_pct = self.clamp(
            raw_json.get("recommended_size_pct", 1.0), lo=0.0, hi=100.0
        )
        ev_estimate = float(raw_json.get("ev_estimate", 0.0))
        max_loss_pct = self.clamp(
            raw_json.get("max_loss_pct", 100.0), lo=0.0, hi=100.0
        )
        edge_durability = max(0.0, float(raw_json.get("edge_durability_hours", 24.0)))
        should_trade = bool(raw_json.get("should_trade", False))
        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "risk_score": risk_score,
            "recommended_size_pct": recommended_size_pct,
            "ev_estimate": ev_estimate,
            "max_loss_pct": max_loss_pct,
            "edge_durability_hours": edge_durability,
            "should_trade": should_trade,
            "reasoning": reasoning,
        }
