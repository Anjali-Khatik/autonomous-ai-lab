"""Fireworks reasoning client.

Used ONLY for reasoning/narrative (interpreting EDA, picking candidate
models, writing plan/verdict/decision narrative). Never used to compute
metrics, counts, or any number that a tool should produce.
"""

import json
import os

from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks

load_dotenv()

DEFAULT_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p6")


def _get_client(model: str) -> ChatFireworks:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set — check .env")
    return ChatFireworks(model=model, api_key=api_key, temperature=0)


def reason(system: str, user: str, response_schema: dict | None = None, model: str = DEFAULT_MODEL) -> dict | str:
    """Call the reasoning model.

    If response_schema is given, instruct JSON-only output matching that
    shape and parse the response into a dict. Otherwise return the raw
    text response. Raises on malformed JSON rather than silently guessing.
    """
    client = _get_client(model)

    messages = [{"role": "system", "content": system}]
    if response_schema is not None:
        schema_instruction = (
            "\n\nRespond with ONLY a single JSON object matching this shape "
            f"(no prose, no markdown fences):\n{json.dumps(response_schema, indent=2)}"
        )
        messages.append({"role": "user", "content": user + schema_instruction})
    else:
        messages.append({"role": "user", "content": user})

    response = client.invoke(messages)
    content = response.content

    if response_schema is None:
        return content

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {content!r}") from e
