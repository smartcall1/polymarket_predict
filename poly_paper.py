#!/usr/bin/env python3
"""
Paper Trader — Polymarket AI Prediction Bot

Uses Gemini Flash to analyze Polymarket prediction markets,
estimates true probabilities vs. market prices, and logs signals.

Usage:
    python paper_trader.py                # Scan once
    python paper_trader.py --settle       # Check settled markets
    python paper_trader.py --dashboard    # Regenerate HTML dashboard
    python paper_trader.py --loop         # Continuous scanning
    python paper_trader.py --stats        # Print stats
"""

import asyncio
import argparse
import os
from datetime import datetime, timezone

from src.paper.tracker import (
    log_signal, settle_signal, take_profit_signal, get_pending_signals, get_all_signals, get_stats,
    has_pending_signal,
)
from src.paper.dashboard import generate_html
from src.config.settings import settings
from src.utils.logging_setup import setup_logging, get_trading_logger

logger = get_trading_logger("paper_trader")

DASHBOARD_OUT = os.path.join(os.path.dirname(__file__), "docs", "paper_dashboard.html")


async def scan_and_log():
    """
    Scan Polymarket markets, run Gemini Flash analysis,
    log actionable signals to paper-trading DB.
    """
    from src.clients.polymarket_client import PolymarketClient
    from src.clients.gemini_client import GeminiClient
    from src.utils.database import DatabaseManager
    from src.jobs.ingest import run_ingestion
    from src.jobs.decide import make_decision_for_market
    from src.utils.telegram import TelegramNotifier

    tg = TelegramNotifier()
    print("[SCAN] Scanning Polymarket markets...")

    poly_client = PolymarketClient()
    db = DatabaseManager()
    await db.initialize()
    gemini = GeminiClient(db_manager=db)

    # Ensemble 모드: ModelRouter 생성 (ENSEMBLE_ENABLED=true일 때 5인방 + Pro 체제)
    from src.clients.model_router import ModelRouter
    model_router = ModelRouter(xai_client=gemini, db_manager=db) if settings.ensemble.enabled else None

    # 1. Ingest markets
    try:
        market_queue: asyncio.Queue = asyncio.Queue()
        await run_ingestion(db, market_queue)

        markets = []
        while not market_queue.empty():
            markets.append(market_queue.get_nowait())

        if not markets:
            print("[SCAN] No eligible markets found.")
            await poly_client.close()
            return 0
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        await poly_client.close()
        return 0

    print(f"[SCAN] {len(markets)} markets to analyze...")

    signals_logged = 0
    analyzed = 0
    skipped = 0

    # 2. Gemini decision on each market
    for i, market in enumerate(markets):
        try:
            print(f"  [{i+1}/{len(markets)}] {market.title[:60]}...", end=" ", flush=True)
            position = await make_decision_for_market(
                market=market,
                db_manager=db,
                gemini_client=gemini,
                poly_client=poly_client,
                model_router=model_router,
            )

            if position is None:
                skipped += 1
                print("SKIP")
                continue

            analyzed += 1
            side = position.side
            confidence = position.confidence or 0
            entry_price = position.entry_price
            rationale = position.rationale or ""

            print(f"-> BUY {side.upper()} (conf={confidence:.0%})")

            if confidence < settings.trading.min_confidence_to_trade:
                continue

            # 중복 시그널 방지: 같은 마켓+사이드에 pending이 이미 있으면 스킵
            if has_pending_signal(market.market_id, side):
                print(f"  [DUP] Already pending: {market.title[:40]} {side.upper()}")
                continue

            signal_id = log_signal(
                market_id=market.market_id,
                market_title=market.title,
                side=side,
                entry_price=entry_price,
                confidence=confidence,
                reasoning=rationale,
                strategy=position.strategy or "directional",
            )
            signals_logged += 1
            print(f"  >>> SIGNAL: BUY {side.upper()} {market.title[:50]} @ {entry_price:.2f} (conf={confidence:.0%})")

            edge = confidence - entry_price if side.upper() == "YES" else confidence - (1 - entry_price)
            tg.notify_signal(
                market_title=market.title, side=side, entry_price=entry_price,
                confidence=confidence, reasoning=rationale, edge=edge,
            )

        except Exception as e:
            print(f"  [ERR] {getattr(market, 'market_id', '?')[:16]}: {e}")
            continue

    await poly_client.close()
    await gemini.close()
    print(f"[DONE] Signals: {signals_logged} | Analyzed: {analyzed} | Skipped: {skipped}")

    tg.notify_scan_complete(signals=signals_logged, skipped=skipped, ai_cost=gemini.total_cost)
    return signals_logged


async def check_settlements():
    """Check Polymarket for settled markets and update signal outcomes."""
    from src.clients.polymarket_client import PolymarketClient
    from src.utils.telegram import TelegramNotifier

    tg = TelegramNotifier()
    pending = get_pending_signals()
    if not pending:
        logger.info("No pending signals to settle.")
        return 0

    client = PolymarketClient()
    settled_count = 0

    for sig in pending:
        try:
            # ── 익절 체크: AI 목표가 도달 시 조기 정산 ──
            entry = sig.get("entry_price", 0)
            confidence = sig.get("confidence", 0)
            side = sig.get("side", "YES").upper()

            effective_conf = confidence if side == "YES" else (1.0 - confidence)
            if entry > 0 and effective_conf > entry:
                ai_target = entry + 0.60 * (effective_conf - entry)

                try:
                    # Polymarket: token_id 없이 conditionId로 가격 조회
                    mkt_check = await client.get_market(sig["market_id"])
                    if mkt_check:
                        prices = client.extract_market_prices(mkt_check)
                        current_yes = prices["yes_price"]
                        current_price = current_yes if side == "YES" else (1.0 - current_yes)

                        if current_price >= ai_target:
                            take_profit_signal(sig["id"], current_price)
                            pnl = current_price - entry
                            logger.info(f"TAKE PROFIT #{sig['id']}: {sig['market_title'][:40]} @ {current_price:.2f} (target={ai_target:.2f}, PnL={pnl:+.2f})")
                            tg.notify_settlement(
                                market_title=sig.get("market_title", ""),
                                side=sig["side"], entry_price=entry,
                                exit_price=current_price, pnl=pnl, result="WIN",
                            )
                            settled_count += 1
                            continue
                        # ── 손절 체크: ROI -15% 이하 시 손절 ──
                        if entry > 0:
                            roi = (current_price - entry) / entry
                            if roi <= -0.15:
                                take_profit_signal(sig["id"], current_price)
                                pnl = current_price - entry
                                logger.info(f"STOP LOSS #{sig['id']}: {sig['market_title'][:40]} @ {current_price:.2f} (ROI={roi:.1%}, PnL={pnl:+.2f})")
                                tg.notify_settlement(
                                    market_title=sig.get("market_title", ""),
                                    side=sig["side"], entry_price=entry,
                                    exit_price=current_price, pnl=pnl, result="LOSS",
                                )
                                settled_count += 1
                                continue

                except Exception as e:
                    logger.debug(f"Price check failed for {sig['market_id']}: {e}")

            # ── 마켓 정산 체크 (기존 로직) ──
            mkt = await client.get_market(sig["market_id"])
            if not mkt:
                continue

            if not client.is_market_resolved(mkt):
                continue

            result = client.get_settlement_result(mkt)
            if result is None:
                continue

            settlement_price = 1.0 if result == "YES" else 0.0

            settle_signal(sig["id"], settlement_price)
            outcome = "WIN" if (
                (sig["side"].upper() == "NO" and settlement_price <= 0.5) or
                (sig["side"].upper() == "YES" and settlement_price >= 0.5)
            ) else "LOSS"

            logger.info(f"Signal #{sig['id']} settled: {outcome} — {sig['market_title'][:50]}")
            settled_count += 1

            entry = sig.get("entry_price", 0)
            pnl = (settlement_price - entry) if sig["side"].upper() == "YES" else (entry - settlement_price)
            tg.notify_settlement(
                market_title=sig.get("market_title", "Unknown"),
                side=sig["side"], entry_price=entry,
                exit_price=settlement_price, pnl=pnl, result=outcome,
            )

        except Exception as e:
            logger.warning(f"Settlement check failed for {sig['market_id'][:16]}: {e}")

    await client.close()
    logger.info(f"Settled {settled_count}/{len(pending)} pending signals")
    return settled_count


def print_stats():
    stats = get_stats()
    print("\n  Polymarket AI Paper Trading Stats")
    print("=" * 45)
    print(f"  Total signals:  {stats['total_signals']}")
    print(f"  Settled:        {stats['settled']}")
    print(f"  Pending:        {stats['pending']}")
    print(f"  Wins:           {stats['wins']}")
    print(f"  Losses:         {stats['losses']}")
    print(f"  Win rate:       {stats['win_rate']}%")
    print(f"  Total P&L:      ${stats['total_pnl']:.2f}")
    print(f"  Avg return:     ${stats['avg_return']:.4f}")
    print(f"  Best trade:     ${stats['best_trade']:.2f}")
    print(f"  Worst trade:    ${stats['worst_trade']:.2f}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Paper Trader — Polymarket AI signal logger")
    parser.add_argument("--settle", action="store_true", help="Check settled markets")
    parser.add_argument("--dashboard", action="store_true", help="Regenerate HTML dashboard only")
    parser.add_argument("--stats", action="store_true", help="Print stats to terminal")
    parser.add_argument("--loop", action="store_true", help="Continuous scanning")
    parser.add_argument("--interval", type=int, default=900, help="Loop interval in seconds (default 15min)")
    args = parser.parse_args()

    setup_logging()

    if args.stats:
        print_stats()
        return

    if args.dashboard:
        os.makedirs(os.path.dirname(DASHBOARD_OUT), exist_ok=True)
        generate_html(DASHBOARD_OUT)
        print(f"Dashboard generated: {DASHBOARD_OUT}")
        return

    if args.settle:
        await check_settlements()
        os.makedirs(os.path.dirname(DASHBOARD_OUT), exist_ok=True)
        generate_html(DASHBOARD_OUT)
        print(f"Dashboard updated: {DASHBOARD_OUT}")
        return

    # Default: scan (+ optionally loop)
    while True:
        await scan_and_log()
        await check_settlements()
        os.makedirs(os.path.dirname(DASHBOARD_OUT), exist_ok=True)
        generate_html(DASHBOARD_OUT)
        logger.info(f"Dashboard updated: {DASHBOARD_OUT}")

        if not args.loop:
            break

        logger.info(f"Sleeping {args.interval}s until next scan...")
        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
