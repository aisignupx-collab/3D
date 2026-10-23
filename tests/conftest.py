import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeLLM:
    """Stands in for LLMClient: returns canned replies and records calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat_json(self, prompt, system="", images=(), retries=2):
        self.calls.append({"prompt": prompt, "system": system, "images": list(images)})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat(self, prompt, system="", images=(), json_mode=False):
        return self.chat_json(prompt, system, images)
