"""
Database manager for the Polymarket AI Prediction Bot.
"""

import aiosqlite
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class Market:
    """Represents a market in the database."""
    market_id: str           # conditionId or Gamma market ID
    title: str
    yes_price: float
    no_price: float
    volume: int
    expiration_ts: int
    category: str
    status: str
    last_updated: datetime
    has_position: bool = False
    description: str = ""
    token_id_yes: str = ""   # CLOB token ID for YES outcome
    token_id_no: str = ""    # CLOB token ID for NO outcome
    slug: str = ""


@dataclass
class Position:
    """Represents a trading position."""
    market_id: str
    side: str
    entry_price: float
    quantity: int
    timestamp: datetime
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    live: bool = False
    status: str = "open"
    id: Optional[int] = None
    strategy: Optional[str] = None
    token_id: str = ""
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_hours: Optional[int] = None
    target_confidence_change: Optional[float] = None


@dataclass
class TradeLog:
    market_id: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    rationale: str
    strategy: Optional[str] = None
    id: Optional[int] = None


class DatabaseManager(TradingLoggerMixin):
    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await db.commit()
        self.logger.info("Database initialized")

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                yes_price REAL, no_price REAL,
                volume INTEGER, expiration_ts INTEGER,
                category TEXT, status TEXT,
                last_updated TEXT,
                description TEXT DEFAULT '',
                token_id_yes TEXT DEFAULT '',
                token_id_no TEXT DEFAULT '',
                slug TEXT DEFAULT ''
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                rationale TEXT, confidence REAL,
                live INTEGER DEFAULT 0,
                status TEXT DEFAULT 'open',
                strategy TEXT,
                token_id TEXT DEFAULT '',
                stop_loss_price REAL,
                take_profit_price REAL,
                max_hold_hours INTEGER,
                target_confidence_change REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT, entry_price REAL, exit_price REAL,
                quantity INTEGER, pnl REAL,
                entry_timestamp TEXT, exit_timestamp TEXT,
                rationale TEXT, strategy TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS market_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                action TEXT, confidence REAL,
                cost REAL, strategy TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS daily_cost_tracking (
                date TEXT PRIMARY KEY,
                total_cost REAL DEFAULT 0,
                request_count INTEGER DEFAULT 0
            )
        """)

    # ── Markets ─────────────────────────────────────

    async def upsert_markets(self, markets: List[Market]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            for m in markets:
                await db.execute("""
                    INSERT INTO markets (market_id, title, yes_price, no_price, volume,
                        expiration_ts, category, status, last_updated, description,
                        token_id_yes, token_id_no, slug)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(market_id) DO UPDATE SET
                        title=excluded.title, yes_price=excluded.yes_price,
                        no_price=excluded.no_price, volume=excluded.volume,
                        expiration_ts=excluded.expiration_ts, status=excluded.status,
                        last_updated=excluded.last_updated, description=excluded.description,
                        token_id_yes=excluded.token_id_yes, token_id_no=excluded.token_id_no,
                        slug=excluded.slug
                """, (m.market_id, m.title, m.yes_price, m.no_price, m.volume,
                      m.expiration_ts, m.category, m.status,
                      m.last_updated.isoformat(), m.description,
                      m.token_id_yes, m.token_id_no, m.slug))
            await db.commit()

    async def get_markets_with_positions(self) -> set:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT DISTINCT market_id FROM positions WHERE status='open'")
            rows = await cursor.fetchall()
            return {r[0] for r in rows}

    # ── Positions ───────────────────────────────────

    async def add_position(self, pos: Position) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO positions (market_id, side, entry_price, quantity, timestamp,
                    rationale, confidence, live, status, strategy, token_id,
                    stop_loss_price, take_profit_price, max_hold_hours, target_confidence_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pos.market_id, pos.side, pos.entry_price, pos.quantity,
                  pos.timestamp.isoformat(), pos.rationale, pos.confidence,
                  1 if pos.live else 0, pos.status, pos.strategy, pos.token_id,
                  pos.stop_loss_price, pos.take_profit_price,
                  pos.max_hold_hours, pos.target_confidence_change))
            await db.commit()
            return cursor.lastrowid

    async def get_open_live_positions(self) -> List[Position]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM positions WHERE status='open' AND live=1")
            rows = await cursor.fetchall()
            return [Position(
                id=r["id"], market_id=r["market_id"], side=r["side"],
                entry_price=r["entry_price"], quantity=r["quantity"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                rationale=r["rationale"], confidence=r["confidence"],
                live=bool(r["live"]), status=r["status"], strategy=r["strategy"],
                token_id=r["token_id"] or "",
                stop_loss_price=r["stop_loss_price"],
                take_profit_price=r["take_profit_price"],
                max_hold_hours=r["max_hold_hours"],
                target_confidence_change=r["target_confidence_change"],
            ) for r in rows]

    async def update_position_to_live(self, position_id: int, fill_price: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE positions SET live=1, entry_price=? WHERE id=?",
                (fill_price, position_id))
            await db.commit()

    async def update_position_status(self, position_id: int, status: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE positions SET status=? WHERE id=?", (status, position_id))
            await db.commit()

    # ── Trade Logs ──────────────────────────────────

    async def add_trade_log(self, log: TradeLog) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO trade_logs (market_id, side, entry_price, exit_price,
                    quantity, pnl, entry_timestamp, exit_timestamp, rationale, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (log.market_id, log.side, log.entry_price, log.exit_price,
                  log.quantity, log.pnl, log.entry_timestamp.isoformat(),
                  log.exit_timestamp.isoformat(), log.rationale, log.strategy))
            await db.commit()

    async def get_all_trade_logs(self) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM trade_logs ORDER BY exit_timestamp DESC")
            return [dict(r) for r in await cursor.fetchall()]

    # ── Cost Tracking ───────────────────────────────

    async def get_daily_ai_cost(self) -> float:
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(cost), 0) FROM market_analyses WHERE date(timestamp)=?",
                (today,))
            row = await cursor.fetchone()
            return row[0] if row else 0.0

    async def record_market_analysis(
        self, market_id: str, action: str, confidence: float,
        cost: float, strategy: str = ""
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO market_analyses (market_id, action, confidence, cost, strategy)
                VALUES (?, ?, ?, ?, ?)
            """, (market_id, action, confidence, cost, strategy))
            await db.commit()

    async def was_recently_analyzed(self, market_id: str, hours: int) -> bool:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM market_analyses WHERE market_id=? AND timestamp>?",
                (market_id, cutoff))
            row = await cursor.fetchone()
            return row[0] > 0 if row else False

    async def get_market_analysis_count_today(self, market_id: str) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM market_analyses WHERE market_id=? AND date(timestamp)=?",
                (market_id, today))
            row = await cursor.fetchone()
            return row[0] if row else 0
