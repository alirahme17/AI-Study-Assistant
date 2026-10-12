"""Offline stand-in for Gemini, for local UI development and demos (FAKE_LLM=1).

It answers by quoting the first sentences of the retrieved sources with proper
[n] citations, so the whole app - streaming, citations, page viewer - can be
exercised without an API key or quota. Never enabled unless FAKE_LLM=1.
"""
import re
import time

from langchain_core.messages import AIMessage, AIMessageChunk

_SOURCE_RE = re.compile(r"^\[(\d+)\] \(([^)]*)\)\n(.*?)(?=\n\n---\n\n\[\d+\] \(|\Z)", re.S | re.M)


class FakeChatModel:
    def __init__(self, delay: float = 0.02):
        self.delay = delay

    @staticmethod
    def _last_user_text(messages) -> str:
        return messages[-1].content if messages else ""

    def invoke(self, messages):
        # Used for follow-up rewriting: return the question unchanged.
        text = self._last_user_text(messages)
        return AIMessage(text.rsplit("Follow-up question:", 1)[-1].strip())

    def _answer(self, messages) -> str:
        prompt = self._last_user_text(messages)
        sources = _SOURCE_RE.findall(prompt)
        if not sources:
            return "I couldn't find this in your course materials."
        parts = ["*(Offline demo answer - FAKE_LLM is on, so this quotes your sources instead of asking Gemini.)*\n"]
        for number, _label, body in sources[:3]:
            sentences = re.split(r"(?<=[.!?])\s", " ".join(body.split()))
            sentence = next((x for x in sentences if len(x) >= 40), sentences[0])
            parts.append(f"{sentence[:280]} [{number}]")
        parts.append("\nFor example, the range follows $R = \\frac{c \\cdot t_d}{2}$ when the delay is $t_d$ [1].")
        return "\n\n".join(parts)

    def stream(self, messages):
        for token in re.findall(r"\S+\s*", self._answer(messages)):
            time.sleep(self.delay)
            yield AIMessageChunk(content=token)
