"""
Trading decision job — analyzes markets and generates trading decisions.
Uses Gemini Flash for AI analysis with cost controls.
"""

import asyncio
import time
import math
from typing import Optional, Dict
from datetime import datetime

from src.utils.database import DatabaseManager, Market, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.gemini_client import GeminiClient, TradingDecision
from src.clients.polymarket_client import PolymarketClient
from src.utils.decision_logger import log_decision

logger = get_trading_logger("decision_engine")


async def _run_ensemble_decision(
    market_data: Dict,
    news_summary: str,
    model_router,
) -> Optional[Dict]:
    """
    Run multi-agent ensemble: 5 Flash agents vote → Trader Pro verifies.
    Returns TradingDecision-compatible dict or None.
    """
    try:
        from src.agents.ensemble import EnsembleRunner
        from src.agents.trader_agent import TraderAgent

        agent_models = settings.ensemble.agent_models

        # Phase 1: Independent ensemble voting
        runner = EnsembleRunner(
            min_models=settings.ensemble.min_models_for_consensus,
            disagreement_threshold=settings.ensemble.disagreement_threshold,
        )

        completions = {}
        for role in runner.agents:
            if role == "trader":
                continue
            role_model = agent_models.get(role, settings.api.gemini_model)

            async def _make_fn(prompt, _model=role_model, _role=role):
                return await model_router.get_completion(
                    prompt=prompt, model=_model,
                    strategy="ensemble", query_type=f"ensemble_{_role}",
                    market_id=market_data.get("ticker"),
                )
            completions[role] = _make_fn

        enriched = {**market_data, "news_summary": news_summary}
        ensemble_result = await runner.run_ensemble(enriched, completions, context={})

        if ensemble_result.get("error") or ensemble_result.get("probability") is None:
            logger.warning(f"Ensemble failed: {ensemble_result.get('error')}")
            return None

        probability = ensemble_result["probability"]
        confidence = ensemble_result["confidence"]
        disagreement = ensemble_result["disagreement"]
        yes_price = float(market_data.get("yes_price", 0.5))

        logger.info(
            f"Ensemble vote: prob={probability:.3f} conf={confidence:.3f} "
            f"disagree={disagreement:.3f} market_yes={yes_price:.2f}"
        )

        # Gate: edge check
        min_edge = settings.trading.min_edge
        edge_yes = probability - yes_price
        edge_no = (1 - probability) - (1 - yes_price)

        if edge_yes >= min_edge:
            suggested_side = "YES"
            edge = edge_yes
        elif edge_no >= min_edge:
            suggested_side = "NO"
            edge = edge_no
        else:
            logger.info(f"Ensemble: no edge (YES={edge_yes:+.3f}, NO={edge_no:+.3f}). SKIP.")
            return None

        if confidence < 0.50:
            logger.info(f"Ensemble: confidence too low ({confidence:.2f}). SKIP.")
            return None

        if disagreement > settings.ensemble.disagreement_threshold:
            logger.info(f"Ensemble: high disagreement ({disagreement:.3f}). SKIP.")
            return None

        # Phase 2: Trader verification (Pro model)
        logger.info(f"Ensemble suggests BUY {suggested_side} (edge={edge:+.3f}). Running Trader (Pro)...")

        trader = TraderAgent()
        trader_model = agent_models.get("trader", "gemini-2.5-pro")

        async def trader_completion(prompt):
            return await model_router.get_completion(
                prompt=prompt, model=trader_model,
                strategy="ensemble_trader", query_type="ensemble_trader",
                market_id=market_data.get("ticker"),
            )

        model_results = ensemble_result.get("model_results", [])
        trader_context = {
            "forecaster_result": next((r for r in model_results if r.get("_agent") == "forecaster"), None),
            "news_result": next((r for r in model_results if r.get("_agent") == "news_analyst"), None),
            "bull_result": next((r for r in model_results if r.get("_agent") == "bull_researcher"), None),
            "bear_result": next((r for r in model_results if r.get("_agent") == "bear_researcher"), None),
            "risk_result": next((r for r in model_results if r.get("_agent") == "risk_manager"), None),
            "ensemble_meta": {
                "probability": probability, "confidence": confidence,
                "disagreement": disagreement, "suggested_side": suggested_side,
                "edge": edge, "num_models": ensemble_result.get("num_models_used", 0),
            },
        }

        trader_result = await trader.analyze(enriched, trader_context, trader_completion)

        if trader_result.get("error"):
            logger.warning(f"Trader verification failed: {trader_result['error']}")
            return None

        action = trader_result.get("action", "SKIP").upper()

        # Trader 결정 로깅
        log_decision(
            market_id=market_data.get("ticker", "?"),
            market_title=market_data.get("title", "?"),
            action=f"TRADER_{action}",
            side=trader_result.get("side", ""),
            yes_price=float(market_data.get("yes_price", 0)),
            confidence=float(trader_result.get("confidence", 0)),
            edge=edge,
            reasoning=trader_result.get("reasoning", "")[:500],
            extra={"model": trader_model, "suggested_side": suggested_side},
        )

        if action in ("BUY", "SELL"):
            logger.info(f"Trader CONFIRMED: {action} {trader_result.get('side')} conf={trader_result.get('confidence'):.2f}")
            return trader_result

        logger.info(f"Trader REJECTED ensemble suggestion (action={action}). SKIP.")
        return None

    except Exception as e:
        logger.error(f"Ensemble decision failed: {e}", exc_info=True)
        return None


def _calculate_dynamic_quantity(
    balance: float, market_price: float, confidence_delta: float,
) -> int:
    """Kelly-based position sizing."""
    if market_price <= 0:
        return 0

    base_pct = settings.trading.default_position_size / 100
    scaler = 1 + (settings.trading.position_size_multiplier * confidence_delta)
    investment = (balance * base_pct) * scaler

    max_investment = (balance * settings.trading.max_position_size_pct) / 100
    final = min(investment, max_investment)
    quantity = int(final // market_price)

    return max(0, quantity)


def estimate_market_volatility(market: Market) -> float:
    """Estimate volatility from price level and volume."""
    try:
        p = getattr(market, "yes_price", 0.5)
        intrinsic_vol = math.sqrt(p * (1 - p))
        vol_factor = max(0.5, min(2.0, 1000 / (market.volume + 100)))
        ttx = get_time_to_expiry_days(market)
        time_factor = max(0.5, min(2.0, math.sqrt(ttx / 7)))
        return max(0.05, min(0.50, intrinsic_vol * vol_factor * time_factor))
    except Exception:
        return 0.15


def get_time_to_expiry_days(market: Market) -> float:
    try:
        if market.expiration_ts:
            return max(0.1, (market.expiration_ts - time.time()) / 86400)
    except Exception:
        pass
    return 7.0


async def make_decision_for_market(
    market: Market,
    db_manager: DatabaseManager,
    gemini_client: GeminiClient,
    poly_client: PolymarketClient,
    model_router=None,
) -> Optional[Position]:
    """Analyze a single market and return a Position if trade is warranted."""
    logger.info(f"Analyzing: {market.title} ({market.market_id[:16]}...)")

    try:
        # CHECK 1: Daily budget
        daily_cost = await db_manager.get_daily_ai_cost()
        if daily_cost >= settings.trading.daily_ai_budget:
            logger.warning(f"Daily AI budget ${settings.trading.daily_ai_budget} exceeded.")
            return None

        # CHECK 2: Recent analysis dedup
        if await db_manager.was_recently_analyzed(market.market_id, settings.trading.analysis_cooldown_hours):
            logger.info(f"Recently analyzed. Skipping.")
            return None

        # CHECK 3: Daily analysis limit
        count_today = await db_manager.get_market_analysis_count_today(market.market_id)
        if count_today >= settings.trading.max_analyses_per_market_per_day:
            logger.info(f"Already analyzed {count_today} times today.")
            return None

        # CHECK 4: Volume threshold for AI
        if market.volume < settings.trading.min_volume_for_ai_analysis:
            return None

        # CHECK 5: Category exclusion
        if market.category.lower() in [c.lower() for c in settings.trading.exclude_low_liquidity_categories]:
            return None

        # Get balance
        balance_resp = await poly_client.get_balance()
        available_balance = float(balance_resp.get("balance", 0))
        portfolio_data = {"available_balance": available_balance}

        total_cost = 0.0

        # Get real-time orderbook prices (CLOB → Gamma fallback)
        yes_price = market.yes_price
        no_price = market.no_price
        has_orderbook = False

        token_id = market.token_id_yes
        if token_id:
            try:
                prices = await poly_client.get_best_prices(token_id)
                if prices and prices.get("mid"):
                    mid = prices["mid"]
                    spread = prices.get("spread")
                    # 오더북이 비어있으면 mid=0.50, spread=None → Gamma 가격 사용
                    if spread is not None and abs(mid - 0.50) > 0.01:
                        yes_price = mid
                        no_price = round(1.0 - mid, 4)
                        has_orderbook = True
                        logger.info(f"Orderbook: YES={yes_price:.2f} NO={no_price:.2f}")
                    elif market.yes_price != 0.5:
                        # Gamma API 가격이 있으면 그걸 사용
                        yes_price = market.yes_price
                        no_price = market.no_price
                        has_orderbook = True
                        logger.info(f"Gamma price (thin orderbook): YES={yes_price:.2f} NO={no_price:.2f}")
                    else:
                        logger.info(f"Both orderbook and Gamma price are 0.50, skipping.")
                        return None
            except Exception as e:
                logger.debug(f"Orderbook fetch failed: {e}")

        # Fallback: Gamma 가격이라도 있으면 사용
        if not has_orderbook and market.yes_price != 0.5:
            yes_price = market.yes_price
            no_price = market.no_price
            has_orderbook = True
            logger.info(f"Gamma only: YES={yes_price:.2f} NO={no_price:.2f}")

        if not has_orderbook:
            logger.info("No price data, skipping.")
            return None
        if yes_price < 0.05 or no_price < 0.05:
            logger.info(f"Dust price (YES={yes_price}), skipping.")
            return None
        if yes_price > 0.95 or yes_price < 0.10:
            logger.info(f"No-edge price (YES={yes_price}), skipping AI analysis.")
            return None

        # Near-expiry low-liquidity filter
        hours_to_expiry = (market.expiration_ts - time.time()) / 3600 if market.expiration_ts else 999
        if hours_to_expiry < 24 and market.volume < 5000:
            logger.info(f"Near-expiry low-liquidity, skipping.")
            return None

        market_data = {
            "ticker": market.market_id,
            "title": market.title,
            "description": market.description,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": market.volume,
            "expiration_ts": market.expiration_ts,
            "category": market.category,
        }

        # News summary (cost-optimized)
        if settings.trading.skip_news_for_low_volume and market.volume < settings.trading.news_search_volume_threshold:
            news_summary = f"Low volume market ({market.volume}). Analysis based on market data only."
        else:
            try:
                news_summary = await asyncio.wait_for(
                    gemini_client.search(market.title, max_length=200), timeout=15.0)
            except Exception:
                news_summary = "News search unavailable."

        # --- Multi-Agent Ensemble Decision (when enabled) ---
        multi_model_ensemble = hasattr(settings, 'ensemble') and settings.ensemble.enabled
        decision = None

        if multi_model_ensemble and model_router:
            logger.info(f"Running multi-agent ensemble for {market.market_id}")
            ensemble_result = await _run_ensemble_decision(
                market_data=market_data,
                news_summary=news_summary,
                model_router=model_router,
            )
            if ensemble_result:
                decision = TradingDecision(
                    action=ensemble_result.get("action", "SKIP"),
                    side=ensemble_result.get("side", "YES"),
                    confidence=float(ensemble_result.get("confidence", 0.0)),
                    limit_price=int(ensemble_result.get("limit_price", 50)) if ensemble_result.get("limit_price") else None,
                )
                decision.reasoning = ensemble_result.get("reasoning", "Multi-agent ensemble decision")
            else:
                logger.info("Ensemble returned no decision (SKIP). Respecting ensemble verdict.")
                await db_manager.record_market_analysis(
                    market.market_id, "ENSEMBLE_SKIP", 0.0, total_cost, "ensemble_no_edge")
                return None

        # --- Fallback: Single-model decision (ensemble 비활성 시에만) ---
        if decision is None:
            decision = await gemini_client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
            )
            total_cost += getattr(decision, "cost", 0.0)

        if not decision:
            await db_manager.record_market_analysis(market.market_id, "SKIP", 0.0, total_cost, "no_decision")
            return None

        logger.info(
            f"Decision: {decision.action} {decision.side} "
            f"conf={decision.confidence:.2f} price={decision.limit_price} "
            f"(cost: ${total_cost:.4f})"
        )

        # Decision log
        edge_val = decision.confidence - (yes_price if decision.side.upper() == "YES" else no_price)
        log_decision(
            market_id=market.market_id,
            market_title=market.title,
            action=decision.action.upper(),
            side=decision.side,
            yes_price=yes_price,
            no_price=no_price,
            confidence=decision.confidence,
            edge=edge_val,
            reasoning=getattr(decision, 'reasoning', '') or '',
            ai_cost=total_cost,
            volume=market.volume,
        )

        await db_manager.record_market_analysis(
            market.market_id, decision.action, decision.confidence, total_cost)

        if decision.action.upper() == "BUY" and decision.confidence >= settings.trading.min_confidence_to_trade:
            price = yes_price if decision.side.upper() == "YES" else no_price

            # Edge filter
            ai_prob = decision.confidence
            market_prob = price
            edge = ai_prob - market_prob
            if abs(edge) < settings.trading.min_edge:
                logger.info(f"Edge too small ({edge:+.3f}), skipping.")
                await db_manager.record_market_analysis(
                    market.market_id, "EDGE_FILTERED", decision.confidence, total_cost)
                return None

            # Position sizing
            confidence_delta = decision.confidence - price
            quantity = _calculate_dynamic_quantity(available_balance, price, confidence_delta)

            if quantity <= 0:
                return None

            # Select token_id based on side
            if decision.side.upper() == "YES":
                tok_id = market.token_id_yes
            else:
                tok_id = market.token_id_no

            rationale = getattr(decision, "reasoning", "No reasoning.")

            # Exit strategy — AI fair value 기반 익절: entry + 60% × (AI추정 - entry)
            # NO side: confidence(YES확률)를 1-confidence(NO fair value)로 변환
            ai_fair_value = decision.confidence if decision.side.upper() == "YES" else 1.0 - decision.confidence
            volatility = estimate_market_volatility(market)
            ttx_days = get_time_to_expiry_days(market)
            stop_loss = max(0.01, price - 0.15 * max(0.5, min(2.0, volatility / 0.1)))
            take_profit = min(0.99, price + 0.60 * (ai_fair_value - price))
            max_hold = min(int(72 * max(0.3, min(3.0, ttx_days / 7))), int(ttx_days * 24 * 0.8))

            position = Position(
                market_id=market.market_id,
                side=decision.side,
                entry_price=price,
                quantity=quantity,
                timestamp=datetime.now(),
                rationale=rationale,
                confidence=decision.confidence,
                live=False,
                strategy="directional",
                token_id=tok_id,
                stop_loss_price=round(stop_loss, 2),
                take_profit_price=round(take_profit, 2),
                max_hold_hours=max(1, max_hold),
            )
            return position

        return None

    except Exception as e:
        logger.error(f"Failed: {market.market_id}: {e}", exc_info=True)
        try:
            await db_manager.record_market_analysis(market.market_id, "ERROR", 0.0, 0.0, "error")
        except Exception:
            pass
        return None
