"""
Polymarket API client for the AI Prediction Bot.

Three API layers:
  - Gamma API  (gamma-api.polymarket.com)  → market metadata, events, settlement
  - CLOB API   (clob.polymarket.com)       → order book, order placement
  - Data API   (data-api.polymarket.com)   → user activity, leaderboard, positions

For paper trading: simulates orders with realistic costs (slippage, fees, gas).
For live trading: uses py-clob-client SDK for real order execution.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any

import httpx

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class PolymarketAPIError(Exception):
    pass


class PolymarketClient(TradingLoggerMixin):
    """
    Polymarket API client for automated trading.
    Handles market data retrieval and trade execution (paper & live).
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.gamma_url = settings.api.gamma_api_url
        self.clob_url = settings.api.clob_api_url
        self.data_url = settings.api.data_api_url
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        self.session = httpx.AsyncClient(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={
                "User-Agent": "PolyPredictBot/1.0",
                "Accept": "application/json",
            },
        )

        # Live trading client (py-clob-client)
        self._clob_client = None
        if settings.trading.live_trading_enabled:
            self._init_clob_client()

        self.logger.info("PolymarketClient initialized",
                         live=settings.trading.live_trading_enabled)

    def _init_clob_client(self):
        """Initialize py-clob-client for live trading."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=settings.api.clob_api_key,
                api_secret=settings.api.clob_api_secret,
                api_passphrase=settings.api.clob_api_passphrase,
            )
            self._clob_client = ClobClient(
                host=self.clob_url,
                key=settings.api.private_key,
                chain_id=137,
                creds=creds,
                signature_type=1,
                funder=settings.api.proxy_address,
            )
            self.logger.info("CLOB client initialized for live trading")
        except Exception as e:
            self.logger.error(f"CLOB client init failed: {e}")

    # ── Generic request with retry ──────────────────

    async def _request(
        self,
        url: str,
        params: Optional[Dict] = None,
        raw: bool = False,
    ) -> Any:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(0.3)
                resp = await self.session.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                return data
            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    wait = self.backoff_factor * (2 ** attempt)
                    self.logger.warning(f"API {e.response.status_code}, retry in {wait:.1f}s")
                    await asyncio.sleep(wait)
                else:
                    raise PolymarketAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                last_exc = e
                await asyncio.sleep(self.backoff_factor * (2 ** attempt))
        raise PolymarketAPIError(f"Failed after {self.max_retries} retries: {last_exc}")

    # ── Gamma API: Markets & Events ─────────────────

    async def get_events(self, limit: int = 100, offset: int = 0, active: bool = True) -> List[Dict]:
        """Get events (market groups) from Gamma API."""
        params = {"limit": limit, "offset": offset, "active": str(active).lower(), "closed": "false"}
        data = await self._request(f"{self.gamma_url}/events", params=params)
        if isinstance(data, list):
            return data
        return data.get("data", data.get("events", []))

    async def get_event(self, slug: str) -> Optional[Dict]:
        """Get single event by slug."""
        data = await self._request(f"{self.gamma_url}/events", params={"slug": slug})
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_markets(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get markets from Gamma API."""
        params = {"limit": limit, "offset": offset, "active": "true", "closed": "false"}
        data = await self._request(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, list):
            return data
        return data.get("data", [])

    async def get_market(self, condition_id: str) -> Optional[Dict]:
        """Get single market by conditionId."""
        data = await self._request(f"{self.gamma_url}/markets", params={"condition_id": condition_id})
        if isinstance(data, list) and data:
            return data[0]
        return None

    async def get_market_by_id(self, market_id: str) -> Optional[Dict]:
        """Get single market by numeric ID or slug."""
        # Try by ID first
        try:
            data = await self._request(f"{self.gamma_url}/markets/{market_id}")
            if data:
                return data
        except Exception:
            pass
        # Try by condition_id
        return await self.get_market(market_id)

    async def get_all_active_events(self, max_pages: int = 10) -> List[Dict]:
        """Paginate through all active events."""
        PAGE_SIZE = 100
        all_events = []
        for page in range(max_pages):
            events = await self.get_events(limit=PAGE_SIZE, offset=page * PAGE_SIZE)
            if not events:
                break
            all_events.extend(events)
            if len(events) < PAGE_SIZE:
                break
            await asyncio.sleep(0.3)
        self.logger.info(f"Fetched {len(all_events)} active events")
        return all_events

    # ── CLOB API: Order Book ────────────────────────

    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Get order book for a token from CLOB API."""
        try:
            data = await self._request(f"{self.clob_url}/book", params={"token_id": token_id})
            return data
        except Exception as e:
            self.logger.debug(f"Orderbook fetch failed for {token_id}: {e}")
            return None

    async def get_best_prices(self, token_id: str) -> Optional[Dict]:
        """Extract best ask/bid/mid from order book."""
        ob = await self.get_orderbook(token_id)
        if not ob:
            return None

        asks = ob.get("asks", [])
        bids = ob.get("bids", [])

        def _best(levels):
            if not levels:
                return None
            lv = levels[0]
            if isinstance(lv, dict):
                return float(lv.get("price", 0)) or None
            return None

        best_ask = _best(asks)
        best_bid = _best(bids)

        if best_ask is None and best_bid is None:
            return None

        return {
            "yes_ask": best_ask,
            "yes_bid": best_bid,
            "spread": round((best_ask or 0) - (best_bid or 0), 4) if best_ask and best_bid else None,
            "mid": round(((best_ask or 0) + (best_bid or 0)) / 2, 4) if best_ask and best_bid else (best_ask or best_bid),
        }

    async def simulate_market_buy_vwap(self, token_id: str, buy_usdc: float) -> Optional[float]:
        """Simulate a market buy and return VWAP (volume-weighted average price)."""
        ob = await self.get_orderbook(token_id)
        if not ob:
            return None

        asks = ob.get("asks", [])
        if not asks:
            return None

        total_spent = 0.0
        total_shares = 0.0
        remaining = buy_usdc

        for level in asks:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            if price <= 0 or size <= 0:
                continue

            level_cost = price * size
            if level_cost <= remaining:
                total_spent += level_cost
                total_shares += size
                remaining -= level_cost
            else:
                shares = remaining / price
                total_spent += remaining
                total_shares += shares
                remaining = 0
                break

        if total_shares <= 0:
            return None
        return round(total_spent / total_shares, 4)

    # ── Market Data Helpers ─────────────────────────

    def extract_market_prices(self, market_data: Dict) -> Dict[str, float]:
        """Extract YES/NO prices from a Gamma API market response."""
        yes_price = 0.5
        no_price = 0.5

        # outcomePrices: "[0.30, 0.70]" or [0.30, 0.70]
        op = market_data.get("outcomePrices")
        if op:
            if isinstance(op, str):
                import json
                try:
                    op = json.loads(op)
                except Exception:
                    op = None
            if isinstance(op, list) and len(op) >= 2:
                yes_price = float(op[0])
                no_price = float(op[1])

        # tokens 배열 fallback
        if yes_price == 0.5:
            tokens = market_data.get("tokens", [])
            for tok in tokens:
                outcome = str(tok.get("outcome", "")).upper()
                price = tok.get("price")
                if price is not None:
                    if outcome == "YES":
                        yes_price = float(price)
                    elif outcome == "NO":
                        no_price = float(price)
            if yes_price != 0.5:
                no_price = round(1.0 - yes_price, 4)

        return {"yes_price": yes_price, "no_price": no_price}

    def get_token_ids(self, market_data: Dict) -> Dict[str, str]:
        """Extract YES/NO token IDs from market data."""
        tokens = market_data.get("tokens", [])
        result = {}
        for tok in tokens:
            outcome = str(tok.get("outcome", "")).upper()
            token_id = tok.get("token_id")
            if token_id:
                result[outcome] = token_id
        # clobTokenIds fallback
        if not result:
            clob_ids = market_data.get("clobTokenIds")
            if clob_ids:
                if isinstance(clob_ids, str):
                    import json
                    try:
                        clob_ids = json.loads(clob_ids)
                    except Exception:
                        clob_ids = None
                if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                    result["YES"] = clob_ids[0]
                    result["NO"] = clob_ids[1]
        return result

    # ── Portfolio (paper mode stubs / live mode) ────

    async def get_balance(self) -> Dict[str, Any]:
        """Get USDC balance."""
        if settings.trading.paper_trading_mode:
            return {"balance": settings.trading.initial_bankroll}

        if self._clob_client:
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                res = self._clob_client.get_balance_allowance(params)
                bal = float(res.get("balance", "0")) / 1_000_000
                return {"balance": bal}
            except Exception as e:
                self.logger.error(f"Balance fetch failed: {e}")
        return {"balance": 0}

    # ── Order Execution ─────────────────────────────

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Dict[str, Any]:
        """
        Place an order. Paper mode: simulate with realistic costs.
        Live mode: use py-clob-client SDK.
        """
        if settings.trading.paper_trading_mode:
            return await self._paper_order(token_id, side, price, size)

        return await self._live_order(token_id, side, price, size)

    async def _paper_order(
        self, token_id: str, side: str, price: float, size: float
    ) -> Dict:
        """Simulate an order with realistic Polymarket costs."""
        actual_price = price
        spread = 0.0

        # 1. 실제 오더북에서 체결가 추정
        try:
            ob = await self.get_orderbook(token_id)
            if ob:
                asks = ob.get("asks", [])
                bids = ob.get("bids", [])

                best_ask = float(asks[0]["price"]) if asks else None
                best_bid = float(bids[0]["price"]) if bids else None

                if side.upper() == "BUY" and best_ask:
                    actual_price = best_ask
                elif side.upper() == "SELL" and best_bid:
                    actual_price = best_bid

                if best_ask and best_bid:
                    spread = round(best_ask - best_bid, 4)
        except Exception:
            pass

        # 2. 슬리피지 (최소 1%, spread 비례)
        base_slippage = max(0.01, spread * 0.5)
        size_slippage = min(0.03, size * 0.0005)
        total_slippage = base_slippage + size_slippage

        if side.upper() == "BUY":
            actual_price = min(actual_price * (1 + total_slippage), 0.99)
        else:
            actual_price = max(actual_price * (1 - total_slippage), 0.01)
        actual_price = round(actual_price, 4)

        # 3. 거래 수수료 (Polymarket: ~0%, but maker/taker may vary)
        fee = 0.0  # Polymarket currently charges 0% trading fee

        # 4. 가스비 (Polygon ~$0.01)
        gas_fee = 0.01

        trade_value = size * actual_price
        total_cost = fee + gas_fee

        self.logger.info(
            f"[PAPER] {side.upper()} {size:.2f} @ {actual_price:.4f} "
            f"(intended={price:.4f}, slip={total_slippage:.1%}, spread={spread:.4f})"
        )

        return {
            "order": {
                "order_id": f"paper_{int(time.time())}_{token_id[:8]}",
                "token_id": token_id,
                "side": side,
                "size": size,
                "price": actual_price,
                "intended_price": price,
                "status": "filled",
                "paper": True,
                "trade_value": round(trade_value, 4),
                "fee": fee,
                "gas_fee": gas_fee,
                "total_cost": round(total_cost, 4),
                "slippage_pct": round(total_slippage * 100, 2),
                "spread": spread,
            }
        }

    async def _live_order(
        self, token_id: str, side: str, price: float, size: float
    ) -> Dict:
        """Place a real order via py-clob-client."""
        if not self._clob_client:
            raise PolymarketAPIError("Live trading: CLOB client not initialized")

        from py_clob_client.order_builder.constants import BUY, SELL
        from py_clob_client.clob_types import OrderArgs

        order_side = BUY if side.upper() == "BUY" else SELL
        safe_price = round(price, 2)
        safe_size = round(size, 2)

        order_args = OrderArgs(
            price=safe_price,
            size=safe_size,
            side=order_side,
            token_id=token_id,
        )

        result = self._clob_client.create_and_post_order(order_args)
        self.logger.info(f"[LIVE] Order placed: {side} {size} @ {price} → {result}")
        return {"order": result}

    async def cancel_order(self, order_id: str) -> Dict:
        return {"status": "cancelled", "order_id": order_id}

    # ── Settlement Helpers ──────────────────────────

    def is_market_resolved(self, market_data: Dict) -> bool:
        """Check if market is resolved."""
        # Check outcomePrices for 0.99+ value
        prices = self.extract_market_prices(market_data)
        if prices["yes_price"] >= 0.99 or prices["no_price"] >= 0.99:
            return True

        # Check explicit fields
        if market_data.get("closed") or market_data.get("resolved"):
            return True

        winner = market_data.get("winnerOutcome")
        if winner:
            return True

        return False

    def get_settlement_result(self, market_data: Dict) -> Optional[str]:
        """Get settlement result: 'YES', 'NO', or None."""
        winner = market_data.get("winnerOutcome")
        if winner:
            return winner.upper()

        prices = self.extract_market_prices(market_data)
        if prices["yes_price"] >= 0.99:
            return "YES"
        if prices["no_price"] >= 0.99:
            return "NO"

        return None

    # ── Lifecycle ────────────────────────────────────

    async def close(self):
        await self.session.aclose()
        self.logger.info("PolymarketClient closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
