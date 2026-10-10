from unittest.mock import MagicMock

from simple_llm import client as cl
from simple_llm.client import CallParams, LLMClient, build_messages


def test_build_messages_with_and_without_system():
    assert build_messages("hi") == [{"role": "user", "content": "hi"}]
    assert build_messages("hi", system="sys") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def _client_with_mock_response(content="ok", stream_chunks=None):
    c = LLMClient("http://localhost:9999/v1", "test-model")
    fake = MagicMock()
    if stream_chunks is None:
        fake.chat.completions.create.return_value.choices[0].message.content = content
    else:
        chunks = []
        for delta in stream_chunks:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = delta
            chunks.append(chunk)
        fake.chat.completions.create.return_value = chunks
    c._client = fake
    return c, fake


def test_complete_passes_params_and_returns_content():
    c, fake = _client_with_mock_response(content="A, B")
    out = c.complete("prompt text", system="sys", params=CallParams(temperature=0.2, max_tokens=8, seed=3))
    assert out == "A, B"
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 8
    assert kwargs["seed"] == 3
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}


def test_complete_stream_yields_deltas():
    c, fake = _client_with_mock_response(stream_chunks=["A", ", ", "B", None])
    out = list(c.complete_stream("p", params=CallParams(max_tokens=4)))
    assert out == ["A", ", ", "B"]  # chunks without content are filtered
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True


def test_default_params_documented_values():
    p = CallParams()
    assert p.temperature == 0.1
