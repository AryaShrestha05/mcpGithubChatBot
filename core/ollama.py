"""Ollama client - drop-in replacement for Claude. Run: ollama pull llama3.2"""

import json
from types import SimpleNamespace

from ollama import Client


class Ollama:
    """Ollama chat client. Same interface as Claude."""

    def __init__(self, model: str):
        self.client = Client()
        self.model = model

    def add_user_message(self, messages: list, message):
        user_message = {"role": "user", "content": message}
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        blocks = []
        for block in message.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            else:
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        messages.append({"role": "assistant", "content": blocks})

    def text_from_message(self, message):
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(self, messages, system=None, temperature=1.0, stop_sequences=None, tools=None, **kwargs):
        ollama_msgs = self._to_ollama_messages(messages)
        ollama_tools = self._to_ollama_tools(tools) if tools else None

        response = self.client.chat(
            model=self.model,
            messages=ollama_msgs,
            tools=ollama_tools,
        )

        return self._to_message(response)

    def _to_ollama_messages(self, messages: list) -> list:
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            result.append({
                                "role": "tool",
                                "tool_name": block.get("tool_name", block.get("tool_use_id", "")),
                                "content": block.get("content", ""),
                            })
                        else:
                            result.append({"role": "user", "content": block.get("text", "")})
                else:
                    result.append({"role": "user", "content": str(content)})

            elif role == "assistant":
                if isinstance(content, list):
                    tool_calls = []
                    text_parts = []
                    for block in content:
                        if block.get("type") == "tool_use":
                            tool_calls.append({
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })
                        else:
                            text_parts.append(block.get("text", ""))
                    if tool_calls:
                        result.append({
                            "role": "assistant",
                            "content": "\n".join(text_parts) or None,
                            "tool_calls": tool_calls,
                        })
                    elif text_parts:
                        result.append({"role": "assistant", "content": "\n".join(text_parts)})
                else:
                    result.append({"role": "assistant", "content": str(content)})
        return result

    def _to_ollama_tools(self, tools: list) -> list:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _to_message(self, response) -> SimpleNamespace:
        msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        if not isinstance(msg, dict):
            msg = {"content": getattr(msg, "content", ""), "tool_calls": getattr(msg, "tool_calls", []) or []}

        content = []
        if msg.get("content"):
            content.append(SimpleNamespace(type="text", text=msg["content"]))
        for i, tc in enumerate(msg.get("tool_calls", [])):
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
            if not isinstance(fn, dict):
                fn = {"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", {})}
            name = fn.get("name", "")
            args = {}
            if "arguments" in fn:
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except json.JSONDecodeError:
                    pass
            content.append(SimpleNamespace(type="tool_use", id=f"call_{name}_{i}", name=name, input=args))

        stop_reason = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop_reason)
