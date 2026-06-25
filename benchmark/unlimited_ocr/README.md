# Unlimited-OCR Debug Harness

This directory is a working area for profiling `baidu/Unlimited-OCR` on
Blackwell-class GPUs with SGLang.

## Server Setup

The Hugging Face model ships a custom SGLang wheel. Install that wheel first:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
uv pip install kernels==0.11.7 pymupdf==1.27.2.2
```

On Blackwell, use FA4 instead of the README's FA3 launch command. FA3 asserts
`SM<=90`; B200/B300 need FA4 or FlashInfer.

```bash
python -m sglang.launch_server \
  --model models/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --attention-backend fa4 \
  --page-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port 10000
```

## Client Benchmark

```bash
python benchmark/unlimited_ocr/bench_unlimited_ocr.py \
  --server-url http://127.0.0.1:10000 \
  --image inputs/synth_doc.png \
  --runs 4 \
  --output traces/fa4_base.json
```

## Current Findings

On a B200 host with the Baidu wheel:

- `fa4` works and is slightly faster than `flashinfer` on a synthetic page.
- SGLang decode CUDA graphs are active in the baseline path.
- Prefill is not captured by CUDA graphs because multimodal image processing and
  prefill shapes remain dynamic.
- `--enable-torch-compile` failed in the Baidu wheel during CUDA graph capture
  because Dynamo traced through `SWAKVPool.set_kv_buffer` and tried to wrap a
  `weakref.proxy` held in `full_to_swa_index_mapping`.
- Mainline SGLang has already refactored this path to pass a `KVWriteLoc` with a
  pre-translated `swa_loc`, so `set_kv_buffer` no longer translates through that
  weakref inside the compiled decode path.

Validated local patch against the Baidu wheel:

```bash
python -m sglang.launch_server \
  --model models/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --attention-backend fa4 \
  --page-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --enable-torch-compile \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --host 0.0.0.0 \
  --port 10000
```

That path improved steady-state end-to-end time on the synthetic page from about
`1.98s` to `1.83s`, with later runs near `1.76s`.

## Next Debug Items

- Test mainline SGLang against `baidu/Unlimited-OCR` once Unlimited-OCR model
  registration is available outside the Baidu wheel.
- Expand compile capture from `--cuda-graph-bs 1` to the production batch-size
  set and isolate any next Dynamo blockers.
- Remove CPU syncs visible in the trace: `aten::nonzero`, `aten::nonzero_numpy`,
  `cudaStreamSynchronize`, `aten::item`, and `index_put_`.
- Make multimodal prefill more static, or separately compile the vision encoder
  and projector path.
- Add B200/B300-specific MoE kernel configs; the server logs reported missing
  configs for `NVIDIA_B200`.
