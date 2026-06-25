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

Additional sampler sync patch:

- The Unlimited-OCR request uses a custom no-repeat n-gram logit processor on
  every decode token.
- The generic custom-processor path recovered request indices with
  `batch_mask.nonzero(as_tuple=True)[0]`, then used boolean indexing. On the
  synthetic bs=1 OCR run this produced 449 `aten::nonzero_numpy` calls and
  about 473 ms of `cudaStreamSynchronize` in the Perfetto trace.
- The patch carries CPU-side custom-processor indices in `SamplingBatchInfo` and
  adds a bs=1 fast path that calls the in-place processor directly.
- With `--enable-torch-compile --cuda-graph-bs 1 --cuda-graph-max-bs 1`, steady
  e2e improved to `1.57s` average over runs 1-3:

```json
{
  "steady_e2e_avg_s": 1.5745927600000869,
  "steady_ttft_avg_s": 0.7232199669997499,
  "chunks": 448,
  "chars": 1659
}
```

With a fixed server seed (`--random-seed 1234`), the original custom-processor
path and the sampler-sync patch produced byte-identical output for the synthetic
page: 1659 chars, SHA-256
`75633291e5fe2eaf07c95bd9f85566dc181230e24552107de26d751a18d8c49a`.

Perfetto operator deltas for one steady profiled request:

| Event | Before | After |
| --- | ---: | ---: |
| `aten::nonzero` | 1352 calls / 585.6 ms | 5 calls / 0.7 ms |
| `aten::nonzero_numpy` | 449 calls / 533.7 ms | 0 calls / 0.0 ms |
| `aten::repeat_interleave` | 449 calls / 36.7 ms | 0 calls / 0.0 ms |
| `cudaStreamSynchronize` | 2256 calls / 473.5 ms | 460 calls / 2.9 ms |
| `cudaGraphLaunch` | 448 calls / 132.2 ms | 448 calls / 115.0 ms |

Local trace artifacts:

- `fa4-steady-1782364683.4941914-TP-0.trace.json.gz`
- `fa4-compile-sampler-patch2-steady-1782366210.3966837-TP-0.trace.json.gz`
- `fa4_compile_sampler_patch2_bs1_bench.json`

## Next Debug Items

- Test mainline SGLang against `baidu/Unlimited-OCR` once Unlimited-OCR model
  registration is available outside the Baidu wheel.
- Expand compile capture from `--cuda-graph-bs 1` to the production batch-size
  set and isolate any next Dynamo blockers.
- The remaining per-token sampler work is mostly greedy `argmax` and
  `index_put_` inside the no-repeat processor. The large `nonzero_numpy`
  synchronization issue is gone for the bs=1 OCR path.
- Make multimodal prefill more static, or separately compile the vision encoder
  and projector path.
- Add B200/B300-specific MoE kernel configs; the server logs reported missing
  configs for `NVIDIA_B200`.
