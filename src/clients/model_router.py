"""
Model router — routes agent requests to the appropriate Gemini model.

Supports role-based model routing: Flash for analysis agents, Pro for Trader.
All requests go through the same Gemini API key, only the model name differs.
"""

import asyncio
import time
from typing import Any, Dict, Optional

from src.clients.gemini_client import GeminiClient, TradingDecision
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class ModelRouter(TradingLoggerMixin):
    """
    Routes AI requests to the appropriate Gemini model.
    Supports per-role model selection for ensemble mode.
    """

    def __init__(
        self,
        xai_client: Optional[Any] = None,
        openrouter_client: Optional[Any] = None,
        db_manager: Any = None,
    ):
        self.db_manager = db_manager
        self.xai_client = xai_client or GeminiClient(db_manager=db_manager)
        self.openrouter_client = openrouter_client  # unused, kept for interface compat

        self.logger.info(
            "ModelRouter initialized (ensemble-capable)",
            default_model=settings.api.gemini_model,
            ensemble_enabled=settings.ensemble.enabled,
        )

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        strategy: Optional[str] = None,
        query_type: Optional[str] = None,
        market_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Get a text completion from the specified Gemini model.
        If model is provided, routes to that model; otherwise uses default Flash.
        """
        if not self.xai_client._client:
            self.logger.warning("Gemini client not initialized, returning empty")
            return ""

        target_model = model or settings.api.gemini_model

        try:
            from google.genai import types

            # Ensemble agents need enough tokens for full reasoning + JSON
            # settings.trading.ai_max_tokens (1024) is too low → reasoning gets truncated
            gen_config = types.GenerateContentConfig(
                temperature=temperature or settings.trading.ai_temperature,
                top_p=0.95,
                max_output_tokens=max_tokens or 8192,
            )

            t0 = time.time()
            response = await asyncio.wait_for(
                self.xai_client._client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=gen_config,
                ),
                timeout=180.0,  # Pro 모델은 긴 컨텍스트 분석 시 시간 필요
            )
            elapsed = time.time() - t0

            # Cost tracking (model-aware)
            usage = getattr(response, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 500) if usage else 500
            out_tok = getattr(usage, "candidates_token_count", 300) if usage else 300
            cost = self.xai_client._estimate_cost(in_tok, out_tok, model=target_model)
            self.xai_client.total_cost += cost
            self.xai_client.request_count += 1
            self.xai_client.daily_tracker.total_cost += cost
            self.xai_client.daily_tracker.request_count += 1
            self.xai_client._save_daily_tracker()

            # Log to DB if available
            if self.db_manager and hasattr(self.db_manager, "log_ai_query"):
                await self.db_manager.log_ai_query(
                    model=target_model,
                    query_type=query_type or "completion",
                    market_id=market_id,
                    cost=cost,
                    response_time=elapsed,
                )

            text = response.text.strip() if response.text else ""
            self.logger.debug(
                f"Completion done [{elapsed:.1f}s, ${cost:.4f}]",
                model=target_model,
                query_type=query_type,
                market_id=market_id,
            )
            return text

        except asyncio.TimeoutError:
            self.logger.error(f"get_completion timed out (model={target_model})")
            return ""
        except Exception as e:
            self.logger.error(f"get_completion failed (model={target_model}): {e}")
            return ""

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
    ) -> Optional[TradingDecision]:
        """Delegate to GeminiClient (single-model legacy path)."""
        return await self.xai_client.get_trading_decision(
            market_data, portfolio_data, news_summary
        )

    async def close(self) -> None:
        """Close all clients."""
        if self.xai_client:
            await self.xai_client.close()
        self.logger.info("ModelRouter closed")
