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
from src.clients.gemini_client import GeminiClient
from src.clients.polymarket_client import PolymarketClient

logger = get_trading_logger("decision_engine")


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

        # Get real-time orderbook prices
        yes_price = market.yes_price
        no_price = market.no_price
        has_orderbook = False

        token_id = market.token_id_yes
        if token_id:
            try:
                prices = await poly_client.get_best_prices(token_id)
                if prices and prices.get("mid"):
                    yes_price = prices["mid"]
                    no_price = round(1.0 - yes_price, 4)
                    has_orderbook = True
                    logger.info(f"Orderbook: YES={yes_price:.2f} NO={no_price:.2f}")
            except Exception as e:
                logger.debug(f"Orderbook fetch failed: {e}")

        # Price filters
        if not has_orderbook:
            logger.info("No orderbook data, skipping.")
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

        # AI decision
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

            # Exit strategy
            volatility = estimate_market_volatility(market)
            ttx_days = get_time_to_expiry_days(market)
            stop_loss = max(0.01, price - 0.15 * max(0.5, min(2.0, volatility / 0.1)))
            take_profit = min(0.99, price + 0.25 * max(0.5, min(2.0, decision.confidence / 0.75)))
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
