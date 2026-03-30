"""
Telegram notifier for Polymarket AI Prediction Bot.
"""

import requests
import logging
from datetime import datetime
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.token = settings.api.telegram_bot_token
        self.chat_id = settings.api.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id
                            and "your_" not in (self.token or "").lower())
        if not self.enabled:
            logger.warning("[Telegram] Token/ChatID missing. Notifications disabled.")

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            chunks = [text[i:i+4090] for i in range(0, len(text), 4090)]
            for chunk in chunks:
                r = requests.post(url, json={
                    "chat_id": self.chat_id, "text": chunk, "parse_mode": parse_mode,
                }, timeout=10)
                r.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"[Telegram] Send failed: {e}")
            return False

    def notify_signal(self, market_title: str, side: str, entry_price: float,
                      confidence: float, reasoning: str, edge: float = 0):
        side_emoji = "\U0001f7e2" if side.upper() == "YES" else "\U0001f534"
        self.send(
            f"{side_emoji} <b>AI Signal: BUY {side.upper()}</b>\n\n"
            f"<b>Market:</b> {self._esc(market_title[:100])}\n"
            f"<b>Entry:</b> {entry_price:.2f} ({entry_price*100:.0f}%)\n"
            f"<b>Confidence:</b> {confidence:.0%}\n"
            f"<b>Edge:</b> {edge:+.1%}\n"
            f"<b>Reasoning:</b> {self._esc(reasoning[:200])}\n\n"
            f"\U0001f4dd <i>Paper Trade (Polymarket AI Bot)</i>"
        )

    def notify_settlement(self, market_title: str, side: str, entry_price: float,
                          exit_price: float, pnl: float, result: str):
        emoji = "\U00002705" if result == "WIN" else "\U0000274c"
        pnl_sign = "+" if pnl >= 0 else ""
        self.send(
            f"{emoji} <b>Settled: {result}</b>\n\n"
            f"<b>Market:</b> {self._esc(market_title[:100])}\n"
            f"<b>Side:</b> {side}\n"
            f"<b>Entry:</b> {entry_price:.2f} -> <b>Exit:</b> {exit_price:.2f}\n"
            f"<b>PnL:</b> {pnl_sign}${pnl:.2f}\n\n"
            f"\U0001f3c1 <i>Polymarket AI Bot</i>"
        )

    def notify_scan_complete(self, signals: int, skipped: int, ai_cost: float):
        self.send(
            f"\U00002705 <b>Scan Complete</b>\n"
            f"Signals: {signals} | Skipped: {skipped}\n"
            f"AI Cost: ${ai_cost:.4f}"
        )

    def notify_daily_summary(self, stats: dict):
        wr = stats.get("win_rate", 0)
        pnl = stats.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        self.send(
            f"\U0001f4ca <b>Daily Summary</b>\n"
            f"{'='*25}\n"
            f"Win Rate: {wr:.1f}%\n"
            f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"Total PnL: {pnl_sign}${pnl:.2f}\n"
            f"AI Cost: ${stats.get('total_ai_cost', 0):.4f}\n"
            f"{'='*25}\n"
            f"<i>Polymarket AI Bot (Paper)</i>"
        )

    def notify_error(self, error_msg: str):
        self.send(f"\U000026a0 <b>Error</b>\n<code>{self._esc(str(error_msg)[:500])}</code>")

    @staticmethod
    def _esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
