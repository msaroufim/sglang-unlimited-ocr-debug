#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible Unlimited-OCR SGLang server.

This is intentionally a small client-side harness. It does not import SGLang,
so it can run from a laptop against a remote server as long as the custom logit
processor string is copied from the server wheel.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


DEEPSEEK_OCR_NO_REPEAT_PROCESSOR = (
    '{"callable": '
    '"80049559000000000000008c2a73676c616e672e7372742e73616d706c696e672e6375'
    '73746f6d5f6c6f6769745f70726f636573736f72948c26446565707365656b4f43524e'
    '6f5265706561744e4772616d4c6f67697450726f636573736f729493942e"}'
)


def encode_image(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix[1:]}"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def run_once(
    session: requests.Session,
    server_url: str,
    model: str,
    image_path: Path,
    image_mode: str,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "document parsing."},
                    encode_image(image_path),
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
        "images_config": {"image_mode": image_mode},
        "custom_logit_processor": DEEPSEEK_OCR_NO_REPEAT_PROCESSOR,
        "custom_params": {"ngram_size": 35, "window_size": 128},
        "stream": stream,
    }

    start = time.perf_counter()
    first_token = None
    chunks = 0
    chars = 0

    response = session.post(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=stream,
        timeout=1200,
    )
    response.raise_for_status()

    if not stream:
        text = response.json()["choices"][0]["message"]["content"]
        first_token = time.perf_counter()
        chunks = 1
        chars = len(text)
    else:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = line[len("data: ") :]
            if event == "[DONE]":
                break
            delta = json.loads(event)["choices"][0].get("delta", {}).get("content", "")
            if not delta:
                continue
            if first_token is None:
                first_token = time.perf_counter()
            chunks += 1
            chars += len(delta)

    end = time.perf_counter()
    return {
        "ttft_s": None if first_token is None else first_token - start,
        "e2e_s": end - start,
        "chunks": chunks,
        "chars": chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:10000")
    parser.add_argument("--model", default="Unlimited-OCR")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-mode", default="gundam")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False

    results = []
    for index in range(args.runs):
        result = run_once(
            session=session,
            server_url=args.server_url,
            model=args.model,
            image_path=args.image,
            image_mode=args.image_mode,
            max_tokens=args.max_tokens,
            stream=not args.no_stream,
        )
        result["run"] = index
        results.append(result)
        print(json.dumps(result), flush=True)

    steady = results[1:] if len(results) > 1 else results
    summary = {
        "runs": results,
        "steady_e2e_avg_s": statistics.mean(r["e2e_s"] for r in steady),
        "steady_ttft_avg_s": statistics.mean(
            r["ttft_s"] for r in steady if r["ttft_s"] is not None
        ),
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
