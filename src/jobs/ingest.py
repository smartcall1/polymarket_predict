"""
Market Ingestion Job — Polymarket

Fetches active events/markets from Polymarket Gamma API,
transforms them into the Market schema, upserts into DB.
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, List

from src.clients.polymarket_client import PolymarketClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


def _parse_polymarket_event(event_data: dict, client: PolymarketClient,
                            existing_position_market_ids: set, logger) -> List[Market]:
    """
    Parse a Polymarket event into Market dataclass instances.

    Polymarket structure: Event → multiple Markets (outcomes).
    각 event.markets[] 항목이 하나의 바이너리 마켓.

    Gamma API 응답 필드:
      - id: "event_id"
      - title: "Will X happen?"
      - slug: "will-x-happen"
      - markets: [{conditionId, outcomePrices, tokens, volume, ...}]
    """
    markets = []
    event_title = event_data.get("title", "Unknown Event")
    event_slug = event_data.get("slug", "")
    event_category = event_data.get("category", "unknown") or "unknown"

    for mkt in event_data.get("markets", []):
        try:
            condition_id = mkt.get("conditionId") or mkt.get("condition_id") or ""
            if not condition_id:
                continue

            # Title — 마켓 자체 question이 있으면 사용, 없으면 event title
            mkt_question = mkt.get("question") or mkt.get("groupItemTitle") or ""
            title = mkt_question if mkt_question else event_title

            # 짧은 title 보강
            if len(title) < 20 and event_title and title != event_title:
                title = f"{event_title}: {title}"

            # Description
            description = mkt.get("description") or event_data.get("description") or ""

            # Prices
            prices = client.extract_market_prices(mkt)
            yes_price = prices["yes_price"]
            no_price = prices["no_price"]

            # Token IDs (CLOB 주문에 필요)
            token_ids = client.get_token_ids(mkt)
            token_id_yes = token_ids.get("YES", "")
            token_id_no = token_ids.get("NO", "")

            # Volume
            vol_raw = mkt.get("volume") or mkt.get("volumeNum") or 0
            volume = int(float(vol_raw))

            # Status
            is_active = mkt.get("active", True)
            is_closed = mkt.get("closed", False)
            if is_closed or not is_active:
                continue  # 비활성 마켓 스킵

            # Expiration
            exp_ts = 0
            exp_raw = mkt.get("endDate") or mkt.get("end_date_iso")
            if exp_raw:
                if isinstance(exp_raw, str):
                    try:
                        exp_ts = int(datetime.fromisoformat(exp_raw.replace("Z", "+00:00")).timestamp())
                    except (ValueError, TypeError):
                        exp_ts = int(time.time()) + 86400 * 30
            if exp_ts == 0:
                exp_ts = int(time.time()) + 86400 * 30

            has_position = condition_id in existing_position_market_ids

            market = Market(
                market_id=condition_id,
                title=title,
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                expiration_ts=exp_ts,
                category=event_category.lower(),
                status="active",
                last_updated=datetime.now(),
                has_position=has_position,
                description=description[:500],
                token_id_yes=token_id_yes,
                token_id_no=token_id_no,
                slug=event_slug or mkt.get("slug", ""),
            )
            markets.append(market)

        except Exception as e:
            logger.error(f"Failed to parse market in event: {e}")
            continue

    return markets


async def process_and_queue_events(
    events: List[dict],
    client: PolymarketClient,
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    existing_position_market_ids: set,
    logger,
):
    """Transform events into markets, upsert to DB, queue eligible ones."""
    all_markets = []
    for event in events:
        parsed = _parse_polymarket_event(event, client, existing_position_market_ids, logger)
        all_markets.extend(parsed)

    if not all_markets:
        logger.info("No active markets found in this batch.")
        return

    await db_manager.upsert_markets(all_markets)
    logger.info(f"Upserted {len(all_markets)} markets from Polymarket.")

    # Category filter
    category_filtered = [
        m for m in all_markets
        if (not settings.trading.preferred_categories
            or m.category in settings.trading.preferred_categories)
        and m.category not in settings.trading.excluded_categories
    ]

    # Volume + price filters
    eligible = []
    for m in category_filtered:
        if m.volume < settings.trading.min_volume:
            continue
        # 토큰 ID 없으면 주문 불가
        if not m.token_id_yes:
            continue
        eligible.append(m)

    logger.info(f"{len(eligible)} eligible markets (vol>=${settings.trading.min_volume}) "
                f"from {len(category_filtered)} category-filtered")

    for market in eligible:
        await queue.put(market)


async def run_ingestion(
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    market_id: Optional[str] = None,
):
    """Main ingestion job — fetches events from Polymarket Gamma API."""
    logger = get_trading_logger("market_ingestion")
    logger.info("Starting Polymarket market ingestion.")

    client = PolymarketClient()

    try:
        existing_position_market_ids = await db_manager.get_markets_with_positions()

        if market_id:
            logger.info(f"Fetching single market: {market_id}")
            mkt = await client.get_market(market_id)
            if mkt:
                # Wrap in event-like structure
                event = {"title": mkt.get("question", ""), "markets": [mkt],
                         "category": "unknown", "slug": mkt.get("slug", "")}
                await process_and_queue_events(
                    [event], client, db_manager, queue,
                    existing_position_market_ids, logger)
        else:
            print("[FETCH] Fetching events from Polymarket Gamma API...")
            all_events = await client.get_all_active_events(max_pages=10)
            print(f"[FETCH] {len(all_events)} events collected")

            await process_and_queue_events(
                all_events, client, db_manager, queue,
                existing_position_market_ids, logger)

    except Exception as e:
        logger.error(f"Error during market ingestion: {e}", exc_info=True)
    finally:
        await client.close()
        logger.info("Market ingestion job finished.")
