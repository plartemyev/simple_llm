"""Integration tests against a running OpenAI-compatible server.

Activated by the `integration` marker; requires:
    SLM_BASE_URL  e.g. http://127.0.0.1:8011/v1
    SLM_MODEL_ID  the served model id (see GET /v1/models)

scripts/integration_test.sh wires both up automatically.
"""

import base64
import json
import os
import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from simple_llm.client import CallParams, LLMClient

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("SLM_BASE_URL", "")
MODEL_ID = os.environ.get("SLM_MODEL_ID", "")


def _red_square_png(size: int = 64) -> bytes:
    # solid-red RGB PNG built with the stdlib only; recognizable enough for a
    # VLM color probe
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    idat = zlib.compress((b"\x00" + b"\xff\x00\x00" * size) * size)  # filter 0 + RGB rows
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


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


def _server_vision(client: LLMClient) -> bool:
    # GET /props (server root) reports the loaded modalities; vision is true
    # only when llama-server runs with a multimodal projector (mmproj)
    root = BASE_URL.rsplit("/v1", 1)[0]
    with urllib.request.urlopen(f"{root}/props", timeout=10) as r:
        return bool(json.load(r).get("modalities", {}).get("vision"))


@requires_server
def test_vision_image_to_text(client):
    try:
        if not _server_vision(client):
            pytest.skip("server runs without an mmproj (text-only)")
    except OSError:
        pytest.skip("/props unreachable")
    data_url = "data:image/png;base64," + base64.b64encode(_red_square_png()).decode()
    resp = client._client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image? One word."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        max_tokens=16,
    )
    out = (resp.choices[0].message.content or "").strip()
    assert out, "empty response for image input"
    assert "red" in out.lower(), f"unexpected answer for a red image: {out!r}"
