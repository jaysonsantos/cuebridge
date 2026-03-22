from __future__ import annotations

from cuebridge import agent
from cuebridge.agent import LangChainSubtitleTranslator
from cuebridge.cancellation import CancellationToken
from langchain_core.messages import AIMessageChunk


class FakeModel:
    def count_input_tokens(self, messages) -> int:
        return len(messages)


def test_langchain_translator_collects_stream_chunks(monkeypatch) -> None:
    class FakeAgent:
        def stream(self, inputs, config, *, stream_mode):
            assert inputs == {"messages": "Hallo"}
            assert stream_mode == "messages"
            assert config["configurable"]["thread_id"] == "thread-1"
            yield (AIMessageChunk(content="Ola "), {})
            yield (AIMessageChunk(content="mundo", chunk_position="last"), {})

        def invoke(self, inputs, config):
            raise AssertionError("translate_text should collect from stream before falling back")

    monkeypatch.setattr(agent, "create_agent", lambda *args, **kwargs: FakeAgent())

    translator = LangChainSubtitleTranslator(
        FakeModel(),
        thread_id="thread-1",
        retain_history=True,
    )

    assert translator.translate_text("Hallo") == "Ola mundo"


def test_langchain_translator_stops_stream_after_cancellation(monkeypatch) -> None:
    class FakeAgent:
        def stream(self, inputs, config, *, stream_mode):
            del inputs, config, stream_mode
            yield (AIMessageChunk(content="Ola "), {})
            yield (AIMessageChunk(content="mundo", chunk_position="last"), {})

        def invoke(self, inputs, config):
            raise AssertionError("translate_text_stream should not fall back to invoke")

    monkeypatch.setattr(agent, "create_agent", lambda *args, **kwargs: FakeAgent())

    translator = LangChainSubtitleTranslator(FakeModel())
    token = CancellationToken()

    stream = translator.translate_text_stream("Hallo", cancellation_token=token)
    first_chunk = next(stream)
    token.cancel("stop now")

    assert first_chunk.text == "Ola "
    assert list(stream) == []
