"""
Gemini Flash client for AI-powered trading decisions.
Uses the google.genai SDK (native async support).
"""

import json
import time
import os
import pickle
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, date

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class TradingDecision:
    """Represents an AI trading decision."""
    action: str          # "buy", "sell", "hold"
    side: str            # "yes", "no"
    confidence: float    # 0.0 to 1.0
    limit_price: Optional[int] = None  # cents (0-99)
    reasoning: Optional[str] = None
    cost: float = 0.0


@dataclass
class DailyUsageTracker:
    date: str
    total_cost: float = 0.0
    request_count: int = 0
    daily_limit: float = 5.0
    is_exhausted: bool = False


ANALYSIS_PROMPT = """You are a team of 5 expert prediction market specialists analyzing a Polymarket market. Execute each role IN ORDER, then produce a final JSON decision.

---
**Market Context:**
- **Title:** {title}
- **Category:** {category}
- **Description:** {description}
- **YES Price:** {yes_price:.2f} (market-implied: {yes_pct:.0f}%)
- **NO Price:** {no_price:.2f} (market-implied: {no_pct:.0f}%)
- **Volume (USD):** ${volume:,.0f}
- **Available Cash:** ${balance:.2f}

**News/Context:**
{news_summary}

---
**STEP 1 — FORECASTER (Base-rate calibrated probability)**
- Start with the BASE RATE: how often do events like this resolve YES historically?
- Update with CURRENT CONDITIONS: specific evidence that shifts probability
- Apply CALIBRATION: guard against overconfidence, regress toward base rate when uncertain
- Output: estimated TRUE probability of YES (0.0 to 1.0)

**STEP 2 — BULL RESEARCHER (Strongest case for YES)**
- Present 3 concrete arguments with evidence for why YES will happen
- Identify CATALYSTS: near-term events that could push probability higher
- Establish PROBABILITY FLOOR: minimum reasonable YES probability

**STEP 3 — BEAR RESEARCHER (Strongest case for NO)**
- Present 3 concrete counter-arguments with evidence for why NO will happen
- Identify RISK FACTORS: what could go wrong for YES holders
- Reference HISTORICAL PRECEDENT: similar events that failed
- Establish PROBABILITY CEILING: maximum reasonable YES probability

**STEP 4 — RISK MANAGER (Quantitative risk/reward)**
- Calculate EV: (estimated_probability - market_price). Require |EV| >= 0.05
- Assess RISK SCORE (1-10): consider volume, time-to-expiry, information quality, bull/bear disagreement
- Check: if bull floor > market price → strong BUY YES signal
- Check: if bear ceiling < market price → strong BUY NO signal
- If risk_score > 7, recommend SKIP regardless of EV

**STEP 5 — TRADER (Final decision)**
Synthesize all analysis into a single JSON decision.
Rules:
- BUY YES only if: estimated_prob > market_yes_price + 0.05 AND confidence >= 0.60
- BUY NO only if: estimated_prob < market_yes_price - 0.05 AND confidence >= 0.60
- SKIP if: edge < 5% OR confidence < 60% OR risk_score > 7 OR agents disagree significantly
- limit_price: target entry in cents (1-99). For BUY YES: near current ask. For BUY NO: 100 - target_no_price
- When in doubt, ALWAYS SKIP. Capital preservation is priority.

**OUTPUT: JSON only, no other text:**
{{"action": "BUY" or "SKIP", "side": "YES" or "NO", "limit_price": 1-99, "confidence": 0.0-1.0, "reasoning": "Include: estimated probability, EV calculation, risk score, key bull/bear factors, and final rationale."}}
"""


class GeminiClient(TradingLoggerMixin):
    """Gemini Flash client for AI-powered trading decisions."""

    def __init__(self, api_key: Optional[str] = None, db_manager=None):
        self.api_key = api_key or settings.api.gemini_api_key
        self.db_manager = db_manager
        self.model_name = settings.api.gemini_model

        self.total_cost = 0.0
        self.request_count = 0
        self.daily_tracker = self._load_daily_tracker()
        self.usage_file = "logs/daily_ai_usage.pkl"

        self._client = None
        self._init_gemini()
        self.logger.info(f"GeminiClient initialized (model={self.model_name})")

    def _init_gemini(self):
        if not self.api_key:
            self.logger.warning("GEMINI_API_KEY not set — dummy mode")
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            self.logger.error(f"Gemini init failed: {e}")

    def _load_daily_tracker(self) -> DailyUsageTracker:
        today_str = date.today().isoformat()
        usage_file = "logs/daily_ai_usage.pkl"
        try:
            if os.path.exists(usage_file):
                with open(usage_file, "rb") as f:
                    tracker = pickle.load(f)
                    if tracker.date == today_str:
                        return tracker
        except Exception:
            pass
        return DailyUsageTracker(date=today_str, daily_limit=settings.trading.daily_ai_cost_limit)

    def _save_daily_tracker(self):
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.usage_file, "wb") as f:
                pickle.dump(self.daily_tracker, f)
        except Exception:
            pass

    async def _check_daily_limits(self) -> bool:
        today_str = date.today().isoformat()
        if self.daily_tracker.date != today_str:
            self.daily_tracker = DailyUsageTracker(date=today_str, daily_limit=settings.trading.daily_ai_cost_limit)
        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            return False
        return True

    def _estimate_cost(self, input_tokens: int = 500, output_tokens: int = 300, model: str = "") -> float:
        model = model or self.model_name
        if "3.1-pro" in model or "3-pro" in model:
            # Gemini 3/3.1 Pro: $2.00 input, $12.00 output
            return (input_tokens / 1_000_000) * 2.00 + (output_tokens / 1_000_000) * 12.00
        if "2.5-pro" in model:
            return (input_tokens / 1_000_000) * 1.25 + (output_tokens / 1_000_000) * 10.00
        # Flash (default)
        return (input_tokens / 1_000_000) * 0.30 + (output_tokens / 1_000_000) * 2.50

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
    ) -> Optional[TradingDecision]:
        """Get a trading decision from Gemini Flash."""
        if not await self._check_daily_limits():
            self.logger.warning("Daily AI cost limit reached. Skipping.")
            return None

        title = market_data.get("title", "Unknown Market")
        category = market_data.get("category", "unknown")
        description = market_data.get("description", "")[:300]

        yes_price = float(market_data.get("yes_price", 0.5))
        no_price = float(market_data.get("no_price", 0.5))
        if yes_price > 1:
            yes_price /= 100.0
            no_price /= 100.0

        volume = market_data.get("volume", 0)
        balance = portfolio_data.get("available_balance", 0)

        prompt = ANALYSIS_PROMPT.format(
            title=title, category=category, description=description,
            yes_price=yes_price, no_price=no_price,
            yes_pct=yes_price * 100, no_pct=no_price * 100,
            volume=volume, balance=balance,
            news_summary=news_summary[:500] if news_summary else "No additional context.",
        )

        if not self._client:
            self.logger.info(f"[DUMMY] Analyzing: {title[:50]}...")
            return TradingDecision(action="hold", side="yes", confidence=0.0)

        try:
            import asyncio
            from google.genai import types

            t0 = time.time()
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=settings.trading.ai_temperature,
                        top_p=0.95,
                        max_output_tokens=32768,
                    ),
                ),
                timeout=60.0,
            )
            elapsed = time.time() - t0

            usage = getattr(response, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 500) if usage else 500
            out_tok = getattr(usage, "candidates_token_count", 300) if usage else 300
            cost = self._estimate_cost(in_tok, out_tok)

            self.total_cost += cost
            self.request_count += 1
            self.daily_tracker.total_cost += cost
            self.daily_tracker.request_count += 1
            self._save_daily_tracker()

            text = response.text.strip()
            result = self._extract_json(text)

            if not result or not isinstance(result, dict):
                self.logger.error(f"JSON extraction failed, raw: {text[:300]}")
                return None

            action = str(result.get("action") or "SKIP").upper()
            side = str(result.get("side") or "YES").upper()
            try:
                confidence = max(0.0, min(1.0, float(result.get("confidence") or 0)))
            except (ValueError, TypeError):
                confidence = 0.0

            raw_price = result.get("limit_price")
            try:
                limit_price = int(raw_price) if raw_price and str(raw_price).strip() not in ("N/A", "None", "null", "") else None
            except (ValueError, TypeError):
                limit_price = None

            reasoning = str(result.get("reasoning") or "")
            td_action = "buy" if action == "BUY" else ("sell" if action == "SELL" else "hold")
            td_side = side.lower()

            self.logger.info(
                f"[Gemini] {title[:50]}... → {td_action.upper()} {td_side.upper()} "
                f"(conf={confidence:.2f}, price={limit_price}) [{elapsed:.1f}s, ${cost:.4f}]"
            )

            return TradingDecision(
                action=td_action, side=td_side, confidence=confidence,
                limit_price=limit_price, reasoning=reasoning, cost=cost,
            )

        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {type(e).__name__}: {e}")
            return None

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from mixed text+JSON Gemini output."""
        import re

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Last ```json block
        blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for block in reversed(blocks):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

        # 3. Last { ... } with "action" key
        matches = re.findall(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 4. Any last { ... }
        matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 5. json-repair fallback
        try:
            from json_repair import repair_json
            cleaned = re.sub(r'```json\s*', '', text)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            if isinstance(repaired, list):
                for item in reversed(repaired):
                    if isinstance(item, dict) and "action" in item:
                        return item
        except Exception:
            pass

        return None

    async def search(self, query: str, max_length: int = 200) -> str:
        return f"Analysis based on market data for: {query[:100]}"

    async def close(self):
        self.logger.info(f"GeminiClient closed. Total: {self.request_count} requests, ${self.total_cost:.4f}")
