import json
import os
from typing import Any, Dict, List, Tuple
import time
from openai import OpenAI
from .tools_local import calc, get_time, summarize
import os
from dotenv import load_dotenv

load_dotenv()   

client = OpenAI()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5-nano")
SMART_MODEL = os.getenv("SMART_MODEL", "gpt-5-mini")

LOCAL_FUNCTION_TOOLS = [
    {
        "type": "function",
        "name": "calc",
        "description": "Safely evaluate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_time",
        "description": "Get the current time in an IANA timezone (e.g., Asia/Dubai).",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "summarize",
        "description": "Convert text lines into a short bullet list (deterministic formatting tool).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "max_bullets": {"type": "integer"},
            },
            "required": ["text", "max_bullets"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

def _exec_local_tool(name: str, args: Dict[str, Any]) -> Any:
    if name == "calc":
        return calc(args["expression"])
    if name == "get_time":
        return get_time(args.get("tz", "Asia/Dubai"))
    if name == "summarize":
        return summarize(args["text"], int(args.get("max_bullets", 5)))
    return {"ok": False, "error": f"Unknown tool: {name}"}

def choose_model(task: str, allow_web_search: bool) -> str:
    complex_markers = ["design", "architecture", "orchestrator", "production", "security", "schema", "agent"]
    if allow_web_search:
        return SMART_MODEL
    if any(m in task.lower() for m in complex_markers):
        return SMART_MODEL
    return DEFAULT_MODEL

def build_tools(allow_web_search: bool) -> List[Dict[str, Any]]:
    tools = list(LOCAL_FUNCTION_TOOLS)
    if allow_web_search:
        tools.append({"type": "web_search"})
    return tools

def run(task: str, allow_web_search: bool, max_hops: int = 8):
    model = choose_model(task, allow_web_search)
    tools = build_tools(allow_web_search)

    resp = client.responses.create(
        model=model,
        tools=tools,
        input=[{"role": "user", "content": task}],
    )

    tool_logs: List[Dict[str, Any]] = []

    for _ in range(max_hops):
        function_calls = [item for item in resp.output if getattr(item, "type", None) == "function_call"]

        if not function_calls:
            return model, (resp.output_text or ""), tool_logs

        tool_output_items: List[Dict[str, Any]] = []

        for fc in function_calls:
            name = fc.name
            args = json.loads(fc.arguments or "{}")
            out = _exec_local_tool(name, args)

            tool_logs.append({"name": name, "arguments": args, "output": out})

            tool_output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(out),
                }
            )

        resp = client.responses.create(
            model=model,
            tools=tools,
            previous_response_id=resp.id,
            input=tool_output_items,
        )

    return model, "Stopped: max tool hops reached.", tool_logs

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_stream(task: str, allow_web_search: bool, max_hops: int = 8):
    model = choose_model(task, allow_web_search)
    tools = build_tools(allow_web_search)

    yield _sse("meta", {"model_used": model, "allow_web_search": allow_web_search})

    resp = client.responses.create(
        model=model,
        tools=tools,
        input=[{"role": "user", "content": task}],
    )

    tool_logs: List[Dict[str, Any]] = []

    for _ in range(max_hops):
        function_calls = [item for item in resp.output if getattr(item, "type", None) == "function_call"]
        if not function_calls:
            break

        tool_output_items: List[Dict[str, Any]] = []

        for fc in function_calls:
            name = fc.name
            args = json.loads(fc.arguments or "{}")
            out = _exec_local_tool(name, args)

            tool_logs.append({"name": name, "arguments": args, "output": out})

            tool_output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(out, ensure_ascii=False),
                }
            )

        resp = client.responses.create(
            model=model,
            tools=tools,
            previous_response_id=resp.id,
            input=tool_output_items,
        )

    yield _sse("tools", {"tool_logs": tool_logs})

    stream = client.responses.create(
        model=model,
        tools=tools,
        previous_response_id=resp.id,
        input=[{"role": "user", "content": "Now provide the final answer."}],
        stream=True,
    )

    try:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield _sse("delta", {"text": event.delta})
            elif event.type == "response.completed":
                yield _sse("done", {"ok": True})
            elif event.type == "error":
                yield _sse("error", {"message": getattr(event, "error", "unknown error")})
                return
    except Exception as e:
        yield _sse("error", {"message": str(e)})
