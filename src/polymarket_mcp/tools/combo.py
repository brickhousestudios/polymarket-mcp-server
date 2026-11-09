"""
Combo Market Discovery Tools for Polymarket MCP Server (BrickHouse addition).

Provides 4 read-only tools for official Polymarket Combos, built on the
PUBLIC Combo RFQ catalog:

- get_combo_markets: Paginated catalog of Combo-eligible markets
- get_combo_market_details: One catalog market, with its leg position IDs
- validate_combo_legs: Local compatibility validation for a set of legs
- analyze_combo: Catalog-only pricing analysis for a proposed set of legs

Authority boundary (by construction, not by policy text):
- This module talks ONLY to the public, unauthenticated Combo catalog at
  combos-rfq-api.polymarket.com. It never imports the auth client or the
  signer, never reads wallet or L2 API credentials, and has no quote,
  order, or RFQ-submission path.
- Catalog outcome prices are market data, NOT executable Combo quotes.
  Executable prices require the authenticated RFQ requester flow, which
  is deliberately not implemented here.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import mcp.types as types

from ..utils import rate_limit_note
from ..utils.rate_limiter import EndpointCategory, RateLimiter, get_rate_limiter

logger = logging.getLogger(__name__)

# Official public Combo catalog (no authentication).
COMBO_API_URL = "https://combos-rfq-api.polymarket.com"
COMBO_MARKETS_PATH = "/v1/rfq/combo-markets"

# Official Combo leg bounds (docs.polymarket.com/trading/combos/requesters).
MIN_COMBO_LEGS = 2
MAX_COMBO_LEGS = 50

# Catalog page bounds for this module: the wire accepts `limit`; we clamp
# to a sane read window and walk pages via `next_cursor` when asked.
_MIN_PAGE_LIMIT = 1
_MAX_PAGE_LIMIT = 100
_MAX_PAGES = 10

_PRICE_NOTE = (
    "Catalog outcome prices are market data only. They are NOT executable "
    "Combo quotes; executable prices require the authenticated RFQ process."
)


async def _note_http_429(
    rate_limiter: RateLimiter,
    response: Any,
    category: EndpointCategory,
) -> None:
    """Thin delegate to the shared 429 note helper (same wiring as the
    Gamma market-fetch sites in market_discovery)."""
    await rate_limit_note.note_http_429(rate_limiter, response, category)


def _clamp_limit(limit: Optional[int]) -> int:
    if limit is None:
        return 20
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    return max(_MIN_PAGE_LIMIT, min(value, _MAX_PAGE_LIMIT))


async def _fetch_combo_page(
    limit: int, cursor: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch ONE page of the public Combo catalog.

    Returns the raw envelope: {"markets": [...], "next_cursor": str|None}.
    Raises on HTTP errors and on malformed (non-dict / markets-not-a-list)
    payloads so callers surface a real error instead of an empty result.
    """
    rate_limiter = get_rate_limiter()
    params: Dict[str, Any] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    async with httpx.AsyncClient(timeout=30.0) as client:
        await rate_limiter.acquire(EndpointCategory.GAMMA_API)
        response = await client.get(
            f"{COMBO_API_URL}{COMBO_MARKETS_PATH}", params=params
        )
        await _note_http_429(rate_limiter, response, EndpointCategory.GAMMA_API)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
        raise ValueError("Combo catalog returned a malformed payload")
    return payload


def _legs_of_market(market: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify the legs of one catalog market: one leg per outcome, each
    carrying its Combo position ID (the identifier an RFQ leg references)."""
    outcomes = market.get("outcomes") or []
    prices = market.get("outcome_prices") or []
    position_ids = market.get("position_ids") or []
    legs = []
    for idx, position_id in enumerate(position_ids):
        legs.append(
            {
                "position_id": str(position_id),
                "outcome": outcomes[idx] if idx < len(outcomes) else None,
                "catalog_price": prices[idx] if idx < len(prices) else None,
                "market_id": market.get("id"),
                "condition_id": market.get("condition_id"),
                "market_title": market.get("title"),
                "market_slug": market.get("slug"),
            }
        )
    return legs


def _normalize_market(market: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": market.get("id"),
        "condition_id": market.get("condition_id"),
        "slug": market.get("slug"),
        "title": market.get("title"),
        "outcomes": market.get("outcomes"),
        "outcome_prices": market.get("outcome_prices"),
        "volume": market.get("volume"),
        "tags": market.get("tags"),
        "pending": market.get("pending"),
        "legs": _legs_of_market(market),
    }


async def get_combo_markets(
    limit: int = 20, cursor: Optional[str] = None, max_pages: int = 1
) -> Dict[str, Any]:
    """Get Combo-eligible markets from the public catalog.

    Walks up to `max_pages` cursor pages (default 1) and returns the
    markets with the cursor for the next page, if any.
    """
    page_limit = _clamp_limit(limit)
    try:
        pages = int(max_pages)
    except (TypeError, ValueError):
        raise ValueError("max_pages must be an integer") from None
    pages = max(1, min(pages, _MAX_PAGES))

    markets: List[Dict[str, Any]] = []
    next_cursor: Optional[str] = cursor
    pages_fetched = 0
    while pages_fetched < pages:
        payload = await _fetch_combo_page(page_limit, next_cursor)
        pages_fetched += 1
        markets.extend(_normalize_market(m) for m in payload["markets"])
        next_cursor = payload.get("next_cursor") or None
        if not next_cursor or not payload["markets"]:
            break

    return {
        "markets": markets,
        "count": len(markets),
        "next_cursor": next_cursor,
        "pages_fetched": pages_fetched,
        "price_note": _PRICE_NOTE,
    }


async def _walk_catalog(max_pages: int = _MAX_PAGES) -> List[Dict[str, Any]]:
    """Fetch catalog pages until the cursor ends (bounded by _MAX_PAGES).
    Used by the lookup/validation tools, which need cross-page search."""
    raw: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(max_pages):
        payload = await _fetch_combo_page(_MAX_PAGE_LIMIT, cursor)
        raw.extend(payload["markets"])
        cursor = payload.get("next_cursor") or None
        if not cursor or not payload["markets"]:
            break
    return raw


async def get_combo_market_details(
    market_id: Optional[str] = None,
    condition_id: Optional[str] = None,
    position_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Find one Combo catalog market by market id, condition id, or one of
    its leg position IDs, searching across catalog pages."""
    if not any([market_id, condition_id, position_id]):
        raise ValueError(
            "Provide one of market_id, condition_id, or position_id"
        )
    for market in await _walk_catalog():
        if market_id is not None and str(market.get("id")) == str(market_id):
            return {"market": _normalize_market(market), "price_note": _PRICE_NOTE}
        if condition_id is not None and market.get("condition_id") == condition_id:
            return {"market": _normalize_market(market), "price_note": _PRICE_NOTE}
        if position_id is not None and str(position_id) in [
            str(p) for p in market.get("position_ids") or []
        ]:
            return {"market": _normalize_market(market), "price_note": _PRICE_NOTE}
    raise ValueError("Combo market not found in the public catalog")


async def _resolve_legs(
    position_ids: List[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Resolve position IDs to catalog legs. Returns (legs, unknown_ids)."""
    wanted = [str(p) for p in position_ids]
    by_position: Dict[str, Dict[str, Any]] = {}
    for market in await _walk_catalog():
        for leg in _legs_of_market(market):
            if leg["position_id"] in wanted and leg["position_id"] not in by_position:
                by_position[leg["position_id"]] = leg
    legs = [by_position[p] for p in wanted if p in by_position]
    unknown = [p for p in wanted if p not in by_position]
    return legs, unknown


def _validate_leg_set(
    position_ids: List[str], legs: List[Dict[str, Any]], unknown: List[str]
) -> Dict[str, Any]:
    """Pure compatibility validation for a proposed Combo leg set."""
    reasons: List[str] = []
    ids = [str(p) for p in position_ids]
    if len(ids) < MIN_COMBO_LEGS:
        reasons.append(
            f"too few legs: {len(ids)} (minimum {MIN_COMBO_LEGS})"
        )
    if len(ids) > MAX_COMBO_LEGS:
        reasons.append(
            f"too many legs: {len(ids)} (maximum {MAX_COMBO_LEGS})"
        )
    if len(set(ids)) != len(ids):
        reasons.append("duplicate position_ids in leg set")
    if unknown:
        reasons.append(f"position_ids not in the public catalog: {unknown}")
    market_ids = [leg.get("market_id") for leg in legs]
    if len(set(market_ids)) != len(market_ids):
        reasons.append("multiple legs from the same market")
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "leg_count": len(ids),
        "legs": legs,
        "unknown_position_ids": unknown,
    }


async def validate_combo_legs(position_ids: List[str]) -> Dict[str, Any]:
    """Validate that a set of position IDs forms a compatible Combo:
    2-50 legs, all present in the public catalog, no duplicates, and no
    two legs from the same market. Local/catalog validation only - this
    does not request a quote and does not guarantee RFQ acceptance."""
    if not isinstance(position_ids, list) or not position_ids:
        raise ValueError("position_ids must be a non-empty list")
    legs, unknown = await _resolve_legs(position_ids)
    result = _validate_leg_set(position_ids, legs, unknown)
    result["price_note"] = _PRICE_NOTE
    return result


async def analyze_combo(position_ids: List[str]) -> Dict[str, Any]:
    """Catalog-only pricing analysis for a proposed Combo: resolves the
    legs, validates compatibility, and reports each leg's catalog outcome
    price plus their arithmetic product (the catalog-implied combined
    price if the legs were independent). NOT an executable quote."""
    validation = await validate_combo_legs(position_ids)
    legs = validation["legs"]
    prices: List[float] = []
    prices_complete = len(legs) == len(position_ids)
    for leg in legs:
        try:
            prices.append(float(leg["catalog_price"]))
        except (TypeError, ValueError):
            prices_complete = False
    implied = None
    if prices_complete and prices:
        implied = 1.0
        for price in prices:
            implied *= price
    return {
        "compatible": validation["compatible"],
        "reasons": validation["reasons"],
        "legs": legs,
        "catalog_prices": prices if prices_complete else None,
        "implied_combined_price": implied,
        "price_note": _PRICE_NOTE,
        "analysis_note": (
            "Implied combined price is the product of catalog outcome "
            "prices (independence arithmetic on market data), not a quote."
        ),
    }


# Tool definitions for MCP
def get_tools() -> List[types.Tool]:
    """Get list of Combo discovery tools (public API, no auth)."""
    return [
        types.Tool(
            name="get_combo_markets",
            description=(
                "List Combo-eligible markets from the official public "
                "Polymarket Combo catalog, with cursor pagination. Each "
                "market includes its leg position IDs. Read-only, no auth. "
                "Catalog prices are not executable quotes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Markets per page (default 20, max 100)",
                        "default": 20,
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor from a previous call",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Pages to walk in this call (default 1, max 10)",
                        "default": 1,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_combo_market_details",
            description=(
                "Find one Combo catalog market by market_id, condition_id, "
                "or a leg position_id, including its identified legs. "
                "Read-only, no auth."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {"type": "string"},
                    "condition_id": {"type": "string"},
                    "position_id": {"type": "string"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="validate_combo_legs",
            description=(
                "Validate a proposed Combo locally against the public "
                "catalog: 2-50 legs, all position IDs in the catalog, no "
                "duplicates, no two legs from the same market. Does not "
                "request a quote."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "position_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Combo leg position IDs (2-50)",
                    }
                },
                "required": ["position_ids"],
            },
        ),
        types.Tool(
            name="analyze_combo",
            description=(
                "Catalog-only pricing analysis for a proposed Combo: leg "
                "resolution, compatibility, catalog outcome prices and "
                "their implied combined price. NOT an executable quote; "
                "executable prices require the authenticated RFQ process."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "position_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Combo leg position IDs (2-50)",
                    }
                },
                "required": ["position_ids"],
            },
        ),
    ]


async def handle_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Handle tool execution.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        List of TextContent with results
    """
    try:
        if name == "get_combo_markets":
            result = await get_combo_markets(**arguments)
        elif name == "get_combo_market_details":
            result = await get_combo_market_details(**arguments)
        elif name == "validate_combo_legs":
            result = await validate_combo_legs(**arguments)
        elif name == "analyze_combo":
            result = await analyze_combo(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Tool execution failed for {name}: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]
