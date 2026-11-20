
import base64
import json
import struct
import threading
import time
import urllib.request
import zlib


def make_png_b64(width: int, height: int, rgb: tuple[int, int, int]) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    r, g, b = rgb
    row = b"\x00" + bytes([r, g, b]) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw, level=1)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def post_json(base_url: str, path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    base_url = "http://10.0.1.71:8000"
    models = json.loads(urllib.request.urlopen(base_url + "/v1/models", timeout=30).read().decode("utf-8"))
    model_id = models["data"][0]["id"]
    results = {}

    # 1. #1036: two concurrent 10-image requests + concurrent text stream
    imgs_a = [{"type": "image_url", "image_url": {"url": make_png_b64(128, 128, (10 + i * 20, 80, 140))}} for i in range(10)]
    imgs_b = [{"type": "image_url", "image_url": {"url": make_png_b64(128, 128, (200 - i * 15, 40, 90))}} for i in range(10)]
    mm_out = [None, None, None]
    mm_err = []

    def worker(idx: int, delay: float, payload: dict) -> None:
        try:
            time.sleep(delay)
            mm_out[idx] = post_json(base_url, "/v1/chat/completions", payload, timeout=300)
        except BaseException as exc:
            mm_err.append((idx, repr(exc)))

    threads = [
        threading.Thread(
            target=worker,
            args=(0, 0.0, {
                "model": model_id,
                "messages": [{"role": "user", "content": [*imgs_a, {"type": "text", "text": "Briefly summarize the dominant colors across these 10 tiles."}]}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }),
        ),
        threading.Thread(
            target=worker,
            args=(1, 0.25, {
                "model": model_id,
                "messages": [{"role": "user", "content": [*imgs_b, {"type": "text", "text": "Are these 10 solid tiles mostly warm or cool? One sentence."}]}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }),
        ),
        threading.Thread(
            target=worker,
            args=(2, 0.15, {
                "model": model_id,
                "messages": [{"role": "user", "content": "Count from 1 to 10 in words."}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }),
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not mm_err, f"#1036 concurrent 10-image error: {mm_err}"
    results["issue_1036_concurrent_2x10_images"] = [
        {
            "content_len": len((r["choices"][0]["message"].get("content") or "")),
            "usage": r.get("usage"),
        }
        for r in mm_out
    ]

    # 2. #943: response_format json_schema with thinking enabled
    r943 = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Return the capital of Japan and its population in millions as JSON."}],
            "max_tokens": 256,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "city_info",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["city", "country"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    msg943 = r943["choices"][0]["message"]
    parsed943 = json.loads(msg943["content"])
    assert "city" in parsed943 and "country" in parsed943, msg943
    results["issue_943_json_schema_thinking"] = {
        "has_reasoning": bool(msg943.get("reasoning") or msg943.get("reasoning_content")),
        "parsed": parsed943,
    }

    # 3. #944: tool_choice="none"
    r944 = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the weather in Paris? Do not call tools, just say you cannot check live weather."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "none",
            "max_tokens": 96,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
        },
    )
    msg944 = r944["choices"][0]["message"]
    assert not msg944.get("tool_calls"), f"#944 returned tool_calls with tool_choice=none: {msg944}"
    results["issue_944_tool_choice_none"] = {
        "tool_calls": msg944.get("tool_calls"),
        "content_preview": (msg944.get("content") or "")[:120],
    }

    # 4. #945: continue_final_message=True, add_generation_prompt=False
    r945 = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Respond with the exact JSON array of the first three prime numbers."},
                {"role": "assistant", "content": "[2, 3, "},
            ],
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"thinking": False},
            "max_tokens": 24,
            "temperature": 0,
        },
    )
    msg945 = r945["choices"][0]["message"]
    txt945 = msg945.get("content") or msg945.get("reasoning") or ""
    assert "5" in txt945, f"#945 unexpected continuation: {r945}"
    results["issue_945_assistant_prefix"] = {
        "content": msg945.get("content"),
        "reasoning": msg945.get("reasoning"),
    }

    # 5. #969: abrupt client disconnect mid-stream followed by clean subsequent request
    import http.client
    conn = http.client.HTTPConnection("10.0.1.71", 8000, timeout=30)
    stream_body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "Write a detailed 500-word essay about ocean currents."}],
        "max_tokens": 256,
        "stream": True,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    })
    conn.request("POST", "/v1/chat/completions", body=stream_body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    first_bytes = resp.read(256)
    conn.close()
    assert len(first_bytes) > 0, "#969 stream returned nothing before disconnect"
    time.sleep(0.5)
    follow = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
        },
    )
    results["issue_969_disconnect_recovery"] = {
        "bytes_before_close": len(first_bytes),
        "followup_content": follow["choices"][0]["message"].get("content"),
    }

    print(json.dumps(results, indent=2), flush=True)


main()
