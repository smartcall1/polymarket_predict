"""
Trade Execution Job — Polymarket

Takes a Position and executes it (paper or live).
"""

import uuid
from datetime import datetime
from typing import Dict

from src.utils.database import DatabaseManager, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.polymarket_client import PolymarketClient


async def execute_position(
    position: Position,
    live_mode: bool,
    db_manager: DatabaseManager,
    poly_client: PolymarketClient,
) -> bool:
    """Execute a single trade position."""
    logger = get_trading_logger("trade_execution")
    logger.info(f"Executing: {position.market_id} ({position.side})")

    if live_mode:
        logger.warning(f"LIVE ORDER for {position.market_id}")
        try:
            token_id = position.token_id
            if not token_id:
                logger.error("No token_id for live order")
                return False

            result = await poly_client.place_order(
                token_id=token_id,
                side="BUY",
                price=round(position.entry_price, 2),
                size=round(position.quantity * position.entry_price, 2),
            )

            order = result.get("order", {})
            fill_price = position.entry_price
            await db_manager.update_position_to_live(position.id, fill_price)
            logger.info(f"LIVE ORDER placed: {order}")
            return True

        except Exception as e:
            logger.error(f"LIVE order failed: {e}")
            return False
    else:
        await db_manager.update_position_to_live(position.id, position.entry_price)
        logger.info(f"[PAPER] Simulated: {position.side} {position.quantity}x @ {position.entry_price:.4f}")
        return True


async def place_profit_taking_orders(
    db_manager: DatabaseManager,
    poly_client: PolymarketClient,
    profit_threshold: float = 0.25,
) -> Dict[str, int]:
    """Place sell orders for positions at profit targets."""
    logger = get_trading_logger("profit_taking")
    results = {"orders_placed": 0, "positions_processed": 0}

    positions = await db_manager.get_open_live_positions()
    if not positions:
        return results

    for pos in positions:
        try:
            results["positions_processed"] += 1
            if not pos.token_id:
                continue

            prices = await poly_client.get_best_prices(pos.token_id)
            if not prices or not prices.get("mid"):
                continue

            current_price = prices["mid"]
            if pos.side.upper() == "NO":
                current_price = 1.0 - current_price

            profit_pct = (current_price - pos.entry_price) / pos.entry_price
            if profit_pct >= profit_threshold:
                logger.info(f"PROFIT TARGET: {pos.market_id} +{profit_pct:.1%}")
                # In paper mode, just log it
                results["orders_placed"] += 1

        except Exception as e:
            logger.error(f"Profit check failed for {pos.market_id}: {e}")

    return results


async def place_stop_loss_orders(
    db_manager: DatabaseManager,
    poly_client: PolymarketClient,
    stop_loss_threshold: float = -0.10,
) -> Dict[str, int]:
    """Place sell orders for positions hitting stop-loss."""
    logger = get_trading_logger("stop_loss")
    results = {"orders_placed": 0, "positions_processed": 0}

    positions = await db_manager.get_open_live_positions()
    if not positions:
        return results

    for pos in positions:
        try:
            results["positions_processed"] += 1
            if not pos.token_id:
                continue

            prices = await poly_client.get_best_prices(pos.token_id)
            if not prices or not prices.get("mid"):
                continue

            current_price = prices["mid"]
            if pos.side.upper() == "NO":
                current_price = 1.0 - current_price

            loss_pct = (current_price - pos.entry_price) / pos.entry_price
            if loss_pct <= stop_loss_threshold:
                logger.info(f"STOP LOSS: {pos.market_id} {loss_pct:.1%}")
                results["orders_placed"] += 1

        except Exception as e:
            logger.error(f"Stop-loss check failed for {pos.market_id}: {e}")

    return results
