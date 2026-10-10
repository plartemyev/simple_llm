"""Integration tests against a running OpenAI-compatible server.

Activated by the `integration` marker; requires:
    SLM_BASE_URL  e.g. http://127.0.0.1:8011/v1
    SLM_MODEL_ID  the served model id (see GET /v1/models)

scripts/integration_test.sh wires both up automatically.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from simple_llm.client import CallParams, LLMClient

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("SLM_BASE_URL", "")
MODEL_ID = os.environ.get("SLM_MODEL_ID", "")

pytest.importorskip("openai")

requires_server = pytest.mark.skipif(
    not (BASE_URL and MODEL_ID), reason="SLM_BASE_URL/SLM_MODEL_ID not set"
)


@pytest.fixture(scope="module")
def client() -> LLMClient:
    return LLMClient(BASE_URL, MODEL_ID, timeout_s=120.0)


@requires_server
def test_basic_completion(client):
    out = client.complete("Reply with exactly one word: OK", params=CallParams(max_tokens=8))
    assert out.strip()
    assert "OK" in out.upper()


@requires_server
def test_usage_and_model_fields(client):
    resp = client._client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=8,
    )
    assert resp.model
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0


@requires_server
def test_streaming_yields_deltas(client):
    deltas = list(client.complete_stream("Count from one to five.", params=CallParams(max_tokens=32)))
    assert any(d.strip() for d in deltas)


@requires_server
def test_determinism_with_seed_and_zero_temperature(client):
    p = CallParams(temperature=0.0, max_tokens=64, seed=42)
    a = client.complete("Repeat: alpha beta gamma delta", params=p)
    b = client.complete("Repeat: alpha beta gamma delta", params=p)
    assert a.strip() == b.strip()


@requires_server
def test_concurrent_completions(client):
    prompts = [f"Say the number {i}" for i in range(4)]

    def call(p):
        return client.complete(p, params=CallParams(max_tokens=16))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(call, prompts))
    wall = time.perf_counter() - t0
    assert all(o.strip() for o in outs)
    # parallel slots should keep total wall below 4x a single slow call
    assert wall < 60
