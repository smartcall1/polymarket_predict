"""
Context Analyst Agent (formerly News Analyst) -- analyses market context
using general knowledge when real-time news is unavailable.

Operates independently in ensemble mode. Provides sentiment and
relevance scores based on known facts about the topic.
"""

from datetime import date

from src.agents.base_agent import BaseAgent


class NewsAnalystAgent(BaseAgent):
    """Analyses market context using general knowledge and available news."""

    AGENT_NAME = "news_analyst"
    AGENT_ROLE = "news_analyst"
    DEFAULT_MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = (
        "You are a prediction market context analyst. Your job is to assess "
        "how background knowledge and available information affect the "
        "probability of a specific market outcome.\n\n"
        "IMPORTANT: You may or may not receive real-time news. If no news "
        "is provided, analyse based on your GENERAL KNOWLEDGE only.\n\n"
        "ANALYSIS FRAMEWORK:\n"
        "1. CATEGORY PATTERNS — What is typical for this category?\n"
        "   - Sports: team form, head-to-head records, injuries\n"
        "   - Crypto: market cycles, sentiment, regulatory trends\n"
        "   - Politics: polling data, historical voting patterns\n"
        "   - Economics: macro indicators, consensus forecasts\n\n"
        "2. KEY FACTORS — List 2-5 concrete factors you are CERTAIN about "
        "that affect this outcome. Only include facts, not speculation.\n\n"
        "3. SEASONAL/CYCLICAL FACTORS — Are there time-based patterns?\n\n"
        "4. INFORMATION QUALITY — How much do you actually know about this "
        "topic? Be honest.\n"
        "   - High relevance (>0.7): you have specific, recent knowledge\n"
        "   - Medium (0.3-0.7): you know the general topic but lack specifics\n"
        "   - Low (<0.3): you know very little — say so clearly\n\n"
        "CRITICAL RULE: If you are not confident about a fact, do NOT include "
        "it. Setting relevance=0.0 when you truly don't know is more valuable "
        "than guessing.\n\n"
        "Return a JSON object (inside a ```json``` code block):\n"
        '  "sentiment": float (-1.0 to 1.0, negative=bearish, positive=bullish),\n'
        '  "relevance": float (0.0 to 1.0, how much you actually know),\n'
        '  "key_factors": list of strings (2-5 verified facts only),\n'
        '  "impact_direction": "up" or "down" or "neutral",\n'
        '  "reasoning": string (your analysis with honesty about knowledge gaps)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        news = market_data.get("news_summary", "")
        category = market_data.get("category", "unknown")
        today = date.today().isoformat()

        if news and "based on market data only" not in news.lower():
            news_section = f"\n\n--- AVAILABLE NEWS ---\n{news[:2000]}\n--- END NEWS ---"
        else:
            news_section = (
                "\n\n[No real-time news available. Analyse using your general "
                "knowledge about this topic. Be honest about what you know "
                "and don't know.]"
            )

        return (
            f"Assess context and sentiment for this prediction market.\n\n"
            f"{summary}\n\n"
            f"Category: {category}\n"
            f"Today's date: {today}\n"
            f"{news_section}\n\n"
            f"If you lack knowledge about this specific topic, set "
            f"relevance=0.0 and say so.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        sentiment = self.clamp(raw_json.get("sentiment", 0.0), lo=-1.0, hi=1.0)
        relevance = self.clamp(raw_json.get("relevance", 0.5))
        key_factors = raw_json.get("key_factors", [])
        if not isinstance(key_factors, list):
            key_factors = [str(key_factors)]
        key_factors = [str(f) for f in key_factors][:10]

        impact = str(raw_json.get("impact_direction", "neutral")).lower()
        if impact not in ("up", "down", "neutral"):
            impact = "neutral"

        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "sentiment": sentiment,
            "relevance": relevance,
            "key_factors": key_factors,
            "impact_direction": impact,
            "reasoning": reasoning,
        }
