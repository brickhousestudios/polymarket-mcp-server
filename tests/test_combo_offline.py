"""Offline tests for polymarket_mcp.tools.combo (BrickHouse addition).

All network access is faked: combo._fetch_combo_page is replaced with a
scripted page source, and the one wire-shape test fakes httpx.AsyncClient.
No credentials are set or needed anywhere in this module's surface.
"""
import json

import httpx
import pytest

from polymarket_mcp.tools import combo


def _market(mid, pids, prices=("0.4", "0.6")):
    return {
        "id": str(mid),
        "condition_id": f"0xcond{mid}",
        "slug": f"market-{mid}",
        "title": f"Market {mid}",
        "outcomes": ["Yes", "No"],
        "outcome_prices": list(prices),
        "position_ids": [str(p) for p in pids],
        "volume": 12345.0,
        "tags": ["test"],
        "pending": False,
    }


PAGES = {
    None: {"markets": [_market(1, (101, 102)), _market(2, (201, 202))],
           "next_cursor": "CUR2"},
    "CUR2": {"markets": [_market(3, (301, 302))], "next_cursor": None},
}


@pytest.fixture
def fake_pages(monkeypatch):
    calls = []

    async def fake_fetch(limit, cursor=None):
        calls.append((limit, cursor))
        return PAGES[cursor]

    monkeypatch.setattr(combo, "_fetch_combo_page", fake_fetch)
    return calls


async def test_get_combo_markets_single_page(fake_pages):
    out = await combo.get_combo_markets(limit=2)
    assert out["count"] == 2
    assert out["next_cursor"] == "CUR2"
    assert out["pages_fetched"] == 1
    first = out["markets"][0]
    assert [leg["position_id"] for leg in first["legs"]] == ["101", "102"]
    assert first["legs"][0]["outcome"] == "Yes"
    assert first["legs"][0]["catalog_price"] == "0.4"
    assert "NOT executable" in out["price_note"]


async def test_get_combo_markets_walks_cursor(fake_pages):
    out = await combo.get_combo_markets(limit=2, max_pages=5)
    assert out["count"] == 3
    assert out["next_cursor"] is None
    assert out["pages_fetched"] == 2
    assert fake_pages == [(2, None), (2, "CUR2")]


async def test_get_combo_market_details_by_position_id(fake_pages):
    out = await combo.get_combo_market_details(position_id="301")
    assert out["market"]["id"] == "3"


async def test_get_combo_market_details_not_found(fake_pages):
    with pytest.raises(ValueError):
        await combo.get_combo_market_details(market_id="999")


async def test_get_combo_market_details_requires_a_key():
    with pytest.raises(ValueError):
        await combo.get_combo_market_details()


async def test_validate_combo_legs_ok(fake_pages):
    out = await combo.validate_combo_legs(["101", "201"])
    assert out["compatible"] is True
    assert out["reasons"] == []
    assert out["leg_count"] == 2


async def test_validate_combo_legs_too_few(fake_pages):
    out = await combo.validate_combo_legs(["101"])
    assert out["compatible"] is False
    assert any("too few" in r for r in out["reasons"])


async def test_validate_combo_legs_duplicate(fake_pages):
    out = await combo.validate_combo_legs(["101", "101"])
    assert out["compatible"] is False
    assert any("duplicate" in r for r in out["reasons"])


async def test_validate_combo_legs_same_market(fake_pages):
    out = await combo.validate_combo_legs(["101", "102"])
    assert out["compatible"] is False
    assert any("same market" in r for r in out["reasons"])


async def test_validate_combo_legs_unknown(fake_pages):
    out = await combo.validate_combo_legs(["101", "999999"])
    assert out["compatible"] is False
    assert out["unknown_position_ids"] == ["999999"]


async def test_validate_combo_legs_too_many(fake_pages):
    out = await combo.validate_combo_legs([str(i) for i in range(51)])
    assert out["compatible"] is False
    assert any("too many" in r for r in out["reasons"])


async def test_validate_combo_legs_rejects_empty():
    with pytest.raises(ValueError):
        await combo.validate_combo_legs([])


async def test_analyze_combo_implied_price(fake_pages):
    out = await combo.analyze_combo(["101", "201"])
    assert out["compatible"] is True
    assert out["catalog_prices"] == [0.4, 0.4]
    assert out["implied_combined_price"] == pytest.approx(0.16)
    assert "NOT executable" in out["price_note"]


async def test_handle_tool_routes_and_unknown(fake_pages):
    out = await combo.handle_tool("get_combo_markets", {"limit": 2})
    payload = json.loads(out[0].text)
    assert payload["count"] == 2
    out = await combo.handle_tool("nope", {})
    payload = json.loads(out[0].text)
    assert "error" in payload


def test_get_tools_shape():
    tools = combo.get_tools()
    names = [t.name for t in tools]
    assert names == [
        "get_combo_markets",
        "get_combo_market_details",
        "validate_combo_legs",
        "analyze_combo",
    ]
    for t in tools:
        assert t.inputSchema["type"] == "object"


def test_module_has_no_auth_surface():
    """The Combo module must not import the auth client or signer."""
    import inspect

    src = inspect.getsource(combo)
    assert "auth.client" not in src
    assert "auth.signer" not in src
    assert "create_polymarket_client" not in src
    assert "POLYGON_PRIVATE_KEY" not in src


class _FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return {"markets": [_market(7, (701, 702))], "next_cursor": "N9"}


class _FakeAsyncClient:
    def __init__(self, *a, **k):
        self.last_url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        self.last_url = url
        self.last_params = params
        return _FakeResponse()


async def test_fetch_combo_page_wire_shape(monkeypatch):
    class _Limiter:
        async def acquire(self, category):
            return None

    monkeypatch.setattr(combo, "get_rate_limiter", lambda: _Limiter())
    monkeypatch.setattr(combo.httpx, "AsyncClient", _FakeAsyncClient)
    out = await combo._fetch_combo_page(5, "CUR")
    assert out["next_cursor"] == "N9"
    assert out["markets"][0]["id"] == "7"


async def test_fetch_combo_page_malformed(monkeypatch):
    class _BadResponse(_FakeResponse):
        def json(self):
            return {"nope": []}

    class _BadClient(_FakeAsyncClient):
        async def get(self, url, params=None):
            return _BadResponse()

    class _Limiter:
        async def acquire(self, category):
            return None

    monkeypatch.setattr(combo, "get_rate_limiter", lambda: _Limiter())
    monkeypatch.setattr(combo.httpx, "AsyncClient", _BadClient)
    with pytest.raises(ValueError):
        await combo._fetch_combo_page(5)


async def test_fetch_combo_page_http_error_raises(monkeypatch):
    class _ErrResponse(_FakeResponse):
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("GET", "https://x"), response=self
            )

    class _ErrClient(_FakeAsyncClient):
        async def get(self, url, params=None):
            return _ErrResponse()

    class _Limiter:
        async def acquire(self, category):
            return None

    monkeypatch.setattr(combo, "get_rate_limiter", lambda: _Limiter())
    monkeypatch.setattr(combo.httpx, "AsyncClient", _ErrClient)
    with pytest.raises(httpx.HTTPStatusError):
        await combo._fetch_combo_page(5)
