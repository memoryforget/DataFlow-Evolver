"""发现工具（网页 / GitHub / arXiv）：只找线索不取数，凭据缺失时按工具降级。

不打真实网络：`requests` 整个换掉，只验证我们这层契约——参数怎么传、结果怎么归一化、
失败怎么说人话、以及"别在这儿下载"这条指引有没有带上。
"""
import asyncio
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dataflow_evolver.llm import mcp_discovery


def _call(tool, **args):
    result = asyncio.run(tool.handler({k: str(v) for k, v in args.items()}))
    return result["content"][0]["text"], result.get("is_error", False)


class _Resp:
    def __init__(self, payload=None, text="", status=200):
        self._payload = payload
        self.text = text
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


def _fake_requests(monkeypatch, *, get=None, post=None):
    fake = types.ModuleType("requests")
    fake.get = get or (lambda *a, **k: _Resp({}))
    fake.post = post or (lambda *a, **k: _Resp({}))
    fake.put = lambda *a, **k: _Resp({})
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    mcp_discovery.configure({"serper_api_key": "k", "github_token": "t"})
    yield
    mcp_discovery.configure({})


# ---- 凭据缺失：按工具降级，不是整台 server 罢工 ----------------------------
def test_web_search_without_key_points_at_alternatives():
    mcp_discovery.configure({})
    text, is_error = _call(mcp_discovery.search_web, query="math dataset")

    assert is_error
    # 少配一个 key 不该让另外两个也用不了，所以要指明还能走哪条路
    assert "serper_api_key" in text and "search_papers" in text


def test_github_works_without_a_token(monkeypatch):
    mcp_discovery.configure({})          # 无 token = 匿名，GitHub 仍允许搜索
    seen = {}

    def _get(url, headers=None, params=None, timeout=None):
        seen["headers"] = headers
        return _Resp({"items": [{"full_name": "a/b", "stargazers_count": 3}]})

    _fake_requests(monkeypatch, get=_get)
    text, is_error = _call(mcp_discovery.search_github, query="math dataset")

    assert not is_error
    assert "Authorization" not in seen["headers"]
    assert json.loads(text)["results"][0]["repo"] == "a/b"


def test_github_token_is_sent_when_configured(monkeypatch):
    seen = {}

    def _get(url, headers=None, params=None, timeout=None):
        seen["headers"] = headers
        return _Resp({"items": []})

    _fake_requests(monkeypatch, get=_get)
    _call(mcp_discovery.search_github, query="q")
    assert seen["headers"]["Authorization"] == "Bearer t"


def test_anonymous_github_failure_mentions_the_rate_limit(monkeypatch):
    mcp_discovery.configure({})

    def _get(*a, **k):
        raise RuntimeError("HTTP 403")

    _fake_requests(monkeypatch, get=_get)
    text, is_error = _call(mcp_discovery.search_github, query="q")
    assert is_error and "限流" in text        # 403 最常见的原因就是匿名限流，直接说破


# ---- 结果归一化 ------------------------------------------------------------
def test_web_search_normalises_serper_results(monkeypatch):
    seen = {}

    def _post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, key=headers.get("X-API-KEY"), body=json)
        return _Resp({"organic": [{"title": "T", "link": "L", "snippet": "S"}]})

    _fake_requests(monkeypatch, post=_post)
    text, _ = _call(mcp_discovery.search_web, query="grade school math corpus", limit=3)

    assert seen["key"] == "k" and seen["body"]["num"] == 3
    assert json.loads(text)["results"] == [{"title": "T", "link": "L", "snippet": "S"}]


def test_arxiv_atom_is_parsed_without_extra_deps(monkeypatch):
    atom = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Synthetic
        Math Data</title>
        <id>http://arxiv.org/abs/1234.5678</id>
        <published>2026-01-01T00:00:00Z</published>
        <summary>  A paper about building corpora.  </summary>
      </entry>
    </feed>"""
    _fake_requests(monkeypatch, get=lambda *a, **k: _Resp(text=atom))

    text, is_error = _call(mcp_discovery.search_papers, query="synthetic math data")
    hit = json.loads(text)["results"][0]

    assert not is_error
    assert hit["title"] == "Synthetic Math Data"      # 标题里的换行要折平
    assert hit["summary"] == "A paper about building corpora."
    assert hit["url"].endswith("1234.5678")


def test_long_abstracts_are_truncated(monkeypatch):
    atom = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        f"<title>T</title><summary>{'x' * 2000}</summary></entry></feed>"
    )
    _fake_requests(monkeypatch, get=lambda *a, **k: _Resp(text=atom))
    hit = json.loads(_call(mcp_discovery.search_papers, query="q")[0])["results"][0]
    assert len(hit["summary"]) < 700 and hit["summary"].endswith("…")


# ---- 分工纪律：发现不取数 --------------------------------------------------
@pytest.mark.parametrize(
    "tool, payload",
    [
        (mcp_discovery.search_github, {"items": [{"full_name": "a/b"}]}),
        (mcp_discovery.search_papers, None),
    ],
)
def test_results_point_back_at_the_download_tools(monkeypatch, tool, payload):
    atom = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>T</title></entry></feed>'
    _fake_requests(monkeypatch, get=lambda *a, **k: _Resp(payload, text=atom))

    text, _ = _call(tool, query="q")

    # 任何真正取数的动作都必须经 dataset-search 落盘留溯源，这里要一路把 Agent 引回去
    next_step = json.loads(text)["next_step"]
    assert "download_dataset" in next_step and "download_ms_dataset" in next_step


def test_empty_query_is_rejected():
    for tool in (mcp_discovery.search_web, mcp_discovery.search_github, mcp_discovery.search_papers):
        _, is_error = _call(tool, query="  ")
        assert is_error


def test_missing_requests_is_explained_not_crashed(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    text, is_error = _call(mcp_discovery.search_papers, query="q")
    assert is_error and "requests" in text
