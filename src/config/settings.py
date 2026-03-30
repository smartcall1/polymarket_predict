"""
Configuration settings for the Polymarket AI Prediction Bot.
Forked from predict.fun_predict → Polymarket + Gemini Flash.
"""

import os
from typing import Dict, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """API configuration — Polymarket + Gemini Flash."""
    # Polymarket CLOB
    clob_api_key: str = field(default_factory=lambda: os.getenv("CLOB_API_KEY", ""))
    clob_api_secret: str = field(default_factory=lambda: os.getenv("CLOB_API_SECRET", ""))
    clob_api_passphrase: str = field(default_factory=lambda: os.getenv("CLOB_API_PASSPHRASE", ""))
    private_key: str = field(default_factory=lambda: os.getenv("PK", ""))
    proxy_address: str = field(default_factory=lambda: os.getenv("POLYMARKET_PROXY_ADDRESS", ""))
    polygon_rpc_url: str = field(default_factory=lambda: os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com"))

    # API URLs
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    data_api_url: str = "https://data-api.polymarket.com"

    # Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing and risk management
    max_position_size_pct: float = 3.0
    max_daily_loss_pct: float = 10.0
    max_positions: int = 15
    min_balance: float = 50.0

    # Market filtering
    min_volume: float = 250000.0     # Polymarket 유동성 풍부 → $250K 이상만 분석 (~1,100개)
    max_time_to_expiry_days: int = 30

    # AI decision thresholds
    min_confidence_to_trade: float = 0.60
    min_edge: float = 0.05
    scan_interval_seconds: int = 300

    # Gemini model config
    ai_temperature: float = 0.1
    ai_max_tokens: int = 1024

    # Kelly Criterion
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25
    max_single_position: float = 0.05
    default_position_size: float = 3.0
    position_size_multiplier: float = 1.0

    # Live/Paper mode
    live_trading_enabled: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true")
    paper_trading_mode: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true")
    initial_bankroll: float = field(default_factory=lambda: float(os.getenv("INITIAL_BANKROLL", "1000.0")))

    # Trading frequency
    max_trades_per_hour: int = 10
    run_interval_minutes: int = 15

    # Category preferences
    preferred_categories: List[str] = field(default_factory=list)
    excluded_categories: List[str] = field(default_factory=lambda: ["esports"])

    # AI cost control
    daily_ai_budget: float = field(default_factory=lambda: float(os.getenv("DAILY_AI_COST_LIMIT", "1.0")))
    daily_ai_cost_limit: float = field(default_factory=lambda: float(os.getenv("DAILY_AI_COST_LIMIT", "1.0")))
    max_ai_cost_per_decision: float = 0.05
    analysis_cooldown_hours: int = 3
    max_analyses_per_market_per_day: int = 4

    # News/sentiment
    skip_news_for_low_volume: bool = True
    news_search_volume_threshold: float = 500.0

    # Category confidence adjustments
    category_confidence_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "sports": 0.90,
        "politics": 1.05,
        "economics": 1.15,
        "crypto": 1.00,
        "default": 1.0,
    })

    # Exclude low liquidity categories
    exclude_low_liquidity_categories: List[str] = field(default_factory=list)
    min_volume_for_ai_analysis: float = 0.0


# === EXIT STRATEGIES ===
profit_threshold: float = 0.20
loss_threshold: float = 0.15
max_hold_time_hours: int = 240

# === MARKET SELECTION ===
max_bid_ask_spread: float = 0.20


@dataclass
class EnsembleConfig:
    """Multi-agent ensemble decision configuration."""
    enabled: bool = field(default_factory=lambda: os.getenv("ENSEMBLE_ENABLED", "false").lower() == "true")
    min_models_for_consensus: int = 3
    disagreement_threshold: float = 0.25
    agent_models: Dict[str, str] = field(default_factory=lambda: {
        "forecaster": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "bull_researcher": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "bear_researcher": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "risk_manager": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "trader": os.getenv("ENSEMBLE_TRADER_MODEL", "gemini-2.5-pro"),
    })


@dataclass
class Settings:
    """Main settings class."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    def validate(self) -> bool:
        if not self.api.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")
        return True


# Global settings instance
settings = Settings()

try:
    settings.validate()
except ValueError as e:
    print(f"[WARN] Configuration: {e}")
    print("Please check your .env file. Copy env.template -> .env and fill in keys.")
