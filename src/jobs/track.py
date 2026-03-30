"""
Position Tracking Job — Polymarket

Monitors open positions for exit triggers:
- Market resolution
- Stop-loss
- Take-profit
- Time-based exits
"""

import asyncio
from datetime import datetime
from typing import Optional

from src.utils.database import DatabaseManager, Position, TradeLog
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.polymarket_client import PolymarketClient


async def should_exit_position(
    position: Position,
    current_yes_price: float,
    current_no_price: float,
    market_resolved: bool,
    market_result: Optional[str] = None,
) -> tuple:
    """
    Determine if position should be exited.
    Returns (should_exit, exit_reason, exit_price).
    """
    current_price = current_yes_price if position.side.upper() == "YES" else current_no_price

    # 1. Market resolution
    if market_resolved:
        if market_result:
            exit_price = 1.0 if market_result.upper() == position.side.upper() else 0.0
        else:
            exit_price = current_price
        return True, "market_resolution", exit_price

    # 2. Stop-loss
    if position.stop_loss_price:
        if position.side.upper() == "YES" and current_price <= position.stop_loss_price:
            return True, "stop_loss", current_price
        if position.side.upper() == "NO" and current_price >= (1.0 - position.stop_loss_price):
            return True, "stop_loss", current_price

    # 3. Take-profit
    if position.take_profit_price:
        if position.side.upper() == "YES" and current_price >= position.take_profit_price:
            return True, "take_profit", current_price
        if position.side.upper() == "NO" and current_price <= (1.0 - position.take_profit_price):
            return True, "take_profit", current_price

    # 4. Time-based exit
    if position.max_hold_hours:
        hours_held = (datetime.now() - position.timestamp).total_seconds() / 3600
        if hours_held >= position.max_hold_hours:
            return True, "time_based", current_price

    # 5. Emergency stop-loss (10%) for positions without explicit stop
    if not position.stop_loss_price:
        loss_pct = (current_price - position.entry_price) / position.entry_price if position.entry_price > 0 else 0
        if loss_pct <= -0.10:
            return True, "emergency_stop_10pct", current_price

    return False, "", current_price


async def run_tracking(db_manager: Optional[DatabaseManager] = None):
    """Enhanced position tracking with exit strategies."""
    logger = get_trading_logger("position_tracking")
    logger.info("Starting position tracking.")

    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()

    poly_client = PolymarketClient()

    try:
        # Step 1: Profit-taking & stop-loss checks
        from src.jobs.execute import place_profit_taking_orders, place_stop_loss_orders

        profit_results = await place_profit_taking_orders(db_manager, poly_client, 0.20)
        stop_results = await place_stop_loss_orders(db_manager, poly_client, -0.15)

        total_orders = profit_results["orders_placed"] + stop_results["orders_placed"]
        if total_orders > 0:
            logger.info(f"Sell orders: profit={profit_results['orders_placed']}, stop={stop_results['orders_placed']}")

        # Step 2: Check all open positions
        positions = await db_manager.get_open_live_positions()
        if not positions:
            logger.info("No open positions to track.")
            return

        logger.info(f"Tracking {len(positions)} open positions.")
        exits = 0

        for pos in positions:
            try:
                # Get current market data from Gamma API
                mkt = await poly_client.get_market(pos.market_id)
                if not mkt:
                    logger.warning(f"Market not found: {pos.market_id}")
                    continue

                prices = poly_client.extract_market_prices(mkt)
                yes_price = prices["yes_price"]
                no_price = prices["no_price"]

                market_resolved = poly_client.is_market_resolved(mkt)
                market_result = poly_client.get_settlement_result(mkt)

                should_exit, reason, exit_price = await should_exit_position(
                    pos, yes_price, no_price, market_resolved, market_result)

                if should_exit:
                    pnl = (exit_price - pos.entry_price) * pos.quantity

                    trade_log = TradeLog(
                        market_id=pos.market_id,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        pnl=pnl,
                        entry_timestamp=pos.timestamp,
                        exit_timestamp=datetime.now(),
                        rationale=f"{pos.rationale or ''} | EXIT: {reason}",
                        strategy=pos.strategy,
                    )

                    await db_manager.add_trade_log(trade_log)
                    await db_manager.update_position_status(pos.id, "closed")
                    exits += 1
                    logger.info(f"Closed {pos.market_id}: {reason} PnL=${pnl:.2f}")
                else:
                    current = yes_price if pos.side.upper() == "YES" else no_price
                    unrealized = (current - pos.entry_price) * pos.quantity
                    hours = (datetime.now() - pos.timestamp).total_seconds() / 3600
                    logger.debug(f"{pos.market_id}: entry={pos.entry_price:.3f} "
                                 f"current={current:.3f} PnL=${unrealized:.2f} {hours:.1f}h")

            except Exception as e:
                logger.error(f"Tracking error for {pos.market_id}: {e}")

        logger.info(f"Tracking done. Exits: {exits}")

    except Exception as e:
        logger.error(f"Tracking job error: {e}", exc_info=True)
    finally:
        await poly_client.close()
