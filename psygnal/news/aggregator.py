"""News-context aggregation: thematic tagging only, never a standalone
signal generator. `NEWS_STATUS = UNAVAILABLE` on any failure."""

from __future__ import annotations

from typing import Any

from psygnal.news.gdelt import extract_articles, fetch_market_news

THEMES: dict[str, tuple[str, ...]] = {
    "usd": ("dollar", "usd", "greenback"),
    "federal_reserve": ("federal reserve", "fed ", "fomc", "powell"),
    "inflation": ("inflation", "cpi", "pce"),
    "interest_rates": ("interest rate", "rate hike", "rate cut", "rate decision"),
    "treasury_yields": ("treasury yield", "bond yield", "10-year"),
    "gold": ("gold", "bullion", "xau"),
    "central_banks": ("central bank", "ecb", "bank of england", "bank of japan", "boe", "boj"),
    "employment": ("jobs report", "payrolls", "unemployment", "labor market"),
    "geopolitics": ("war", "conflict", "sanctions", "geopolitical"),
    "oil": ("oil", "crude", "opec"),
    "risk_sentiment": ("risk-on", "risk-off", "risk appetite", "safe haven"),
}


def _tag_themes(title: str) -> list[str]:
    title_lower = title.lower()
    return [theme for theme, keywords in THEMES.items() if any(kw in title_lower for kw in keywords)]


def summarize_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    theme_counts: dict[str, int] = {theme: 0 for theme in THEMES}
    for article in articles:
        title = article.get("title", "")
        for theme in _tag_themes(title):
            theme_counts[theme] += 1

    active_themes = {k: v for k, v in theme_counts.items() if v > 0}
    dominant = sorted(active_themes.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "article_count": len(articles),
        "theme_counts": theme_counts,
        "dominant_themes": [t for t, _ in dominant],
    }


def get_news_state(enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"status": "DISABLED", "article_count": 0, "dominant_themes": []}

    result = fetch_market_news()
    if not result.ok:
        return {"status": "UNAVAILABLE", "article_count": 0, "dominant_themes": [], "fetch_error": result.error}

    articles = extract_articles(result.data)
    summary = summarize_articles(articles)
    summary["status"] = "OK" if articles else "UNAVAILABLE"
    summary["from_cache"] = result.from_cache
    return summary
