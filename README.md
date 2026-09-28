# Question: is `--kv-stream-arena-mib 3072` at 262144 ctx inside the supported range on a 24 GiB card?

Evidence bundle for a question addressed to
[`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming),
branch `feature/kv-stream-phase-arena`, commit
[`d7680c6`](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming/commit/d7680c67b89f7cd4e3069b4bc2f26747aa588e3d).

This is a clarification request, not a bug report. I ran the phase-arena branch on
hardware it is explicitly not tuned for (RTX 3090 24 GiB, vs the documented RTX 5070
Ti 16 GiB) and picked an arena size by hand. I would like to know whether the number
I picked was outside what you support before I read anything into the result.

## What I observed

With `--kv-stream-arena-mib 3072` at `--ctx-size 262144`, `llama-server` emits a
**different token sequence** than the same binary, same prompt, same sampler with
streaming off. At `temperature 0` with `cache_prompt` off, exact attention over the
full context should give bit-identical token IDs. It diverges at token index 4
(synthetic prompt) / index 2 (prose prompt), and the synthetic case degenerates into
a 253-token `"The"` run.

It is **not** arena-size-monotonic in the obvious way: arena 6144 at the same context
is token-identical to the control, and arena 1024 at 131072 ctx — which thrashes the
ring *harder* by copy-busy % — is also identical. Divergence appears only in the
262144/3072 cell.

## Hardware / model

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, driver 595.91.07 (single GPU, exclusive) |
| Model | `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13,146,393,504 bytes) |
| Reported version string | `0.4.0-dev (build 1, commit d7680c6)`, GNU 12.4.0, Linux x86_64 |
| Idle VRAM before each arm | ~1195 MiB (pre-existing non-benchmark residency) |

## Reproducer

### Server (the failing cell)

```sh
/opt/akv/llama-server \
  -m /opt/akv/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --alias akv-bench --host 127.0.0.1 --port 12355 \
  --ctx-size 262144 \
  -fa on -ctk q8_0 -ctv q4_0 -ngl all \
  -b 256 -ub 256 -np 1 \
  --no-mmproj --no-warmup --fit off --reasoning-format none \
  --kv-stream-arena-mib 3072
```

The control is the **same command with `--kv-stream-arena-mib` omitted entirely**.
Everything else — binary, model, context size, cache quant, batch sizes, prompt,
sampler — is held fixed. That is the only difference between the two arms.

### Prompt A — synthetic repetitive

Prefix is a single token ID repeated, then a short natural suffix. Total prompt
length is `ctx - 256 = 261888` tokens, so the 256 decoded tokens fit without
eviction of the tail.

```
prompt_tokens  = [760] * (261888 - len(S)) + S
where S        = tokenize("The capital of France is")   # via the server's /tokenize
```

Token 760 is an arbitrary fixed ID; the suffix makes the correct continuation
obvious (`" Paris."`) so a wrong answer is legible.

### Prompt B — non-repetitive prose

Same shape, but the prefix is real prose instead of one repeated token, to rule out
prompt-driven degeneration as the explanation:

```
prose_tokens   = tokenize(first 1,600,000 chars of Project Gutenberg "War and Peace")
                 # 1,599,965 chars -> 397,680 tokens, no tiling needed
prompt_tokens  = prose_tokens[:261888 - len(S)] + S
```

### Completion request (identical for every arm)

```
POST http://127.0.0.1:12355/completion
{
  "prompt":        <prompt_tokens as an array of ints>,
  "n_predict":     256,
  "ignore_eos":    true,
  "cache_prompt":  false,
  "temperature":   0,
  "return_tokens": true
}
```

Compare the returned token ID arrays between the streaming arm and its control.
Equality of IDs — not coherence of the decoded text — is the test.

## Results matrix

| ctx | arena MiB | prompt | token-identical to control? | first diverging index | ring miss % | resident pages/layer | ring slots | copy busy % | TG tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 262144 | 0 (off) | synthetic | control | — | — | — | — | — | 14.62 |
| **262144** | **3072** | **synthetic** | **NO — DIVERGED** | **4** | **97.3** | **471 → 430** | **8 → 664** | **82.3** | **3.11** |
| 262144 | 6144 | synthetic | yes | — | 72.7 | 943 → 938 | 18 → 98 | 43.4 | 12.89 |
| 262144 | 12288 | synthetic | n/a — OOM at startup | — | — | — | — | — | — |
| 262144 | 0 (off) | prose | control | — | — | — | — | — | 14.60 |
| **262144** | **3072** | **prose** | **NO — DIVERGED** | **2** | **97.3** | **471 → 430** | **8 → 664** | **82.4** | **3.10** |
| 131072 | 0 (off) | synthetic | control | — | — | — | — | — | 22.19 |
| 131072 | 3072 | synthetic | yes | — | 72.2 | 471 → 468 | 8 → 56 | 35.6 | 19.62 |
| 131072 | 1024 | synthetic | yes | — | 88.5 | 155 → 130 | 23 → 423 | 85.1 | 4.82 |

`resident pages/layer`, `ring slots`, `miss %` and `copy busy %` are read from the
last `kv_stream_adapt:` line in each arm's server log, e.g. for the failing cell:

```
W kv_stream_adapt: adaptive KV partition: resident pages/layer 471 -> 430, ring slots 8 -> 664, miss 97.3%, copy busy 82.3%
```

The 262144/12288 arm could not start:

```
E ggml_backend_cuda_phase_arena_new: allocating 12288.00 MiB phase arena on device 0 failed: out of memory
E llama_init_from_model: failed to initialize the context: failed to create CUDA block KV streaming phase arena
```

## Determinism

The failing cell is **not** flaky — it is reproducibly wrong the same way. Three
repeats of 262144/3072/synthetic were run back to back; all three produced
**bit-identical token sequences to each other**, and all three diverged from the
control at the same index 4:

```
       check    verdict  firstdiff  A vs B
 determinism  IDENTICAL          -  ctx262144-arena0-token-r1  vs  ctx262144-arena0-token-r2
 determinism  IDENTICAL          -  ctx262144-arena3072-token-r1  vs  ctx262144-arena3072-token-r2
 determinism  IDENTICAL          -  ctx262144-arena3072-token-r1  vs  ctx262144-arena3072-token-r3
   exactness   DIVERGED          4  ctx262144-arena0-token-r1  vs  ctx262144-arena3072-token-r1
   exactness   DIVERGED          4  ctx262144-arena0-token-r1  vs  ctx262144-arena3072-token-r2
   exactness   DIVERGED          4  ctx262144-arena0-token-r1  vs  ctx262144-arena3072-token-r3
   exactness  IDENTICAL          -  ctx262144-arena0-token-r1  vs  ctx262144-arena6144-token-r1
   exactness   DIVERGED          2  ctx262144-arena0-prose-r1  vs  ctx262144-arena3072-prose-r1
   exactness  IDENTICAL          -  ctx131072-arena0-token-r1  vs  ctx131072-arena3072-token-r1
   exactness  IDENTICAL          -  ctx131072-arena0-token-r1  vs  ctx131072-arena1024-token-r1
```

The control is also self-consistent (2/2 identical), so the comparison base is sound.

### Decoded text, for illustration

Synthetic prompt, 262144, arena 0 (control):

```
 Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris. [...]
```

Synthetic prompt, 262144, arena 3072:

```
 Paris.\n\nTheTheTheTheTheTheTheTheTheThe [...]   (253-token immediate repeat)
```

Prose prompt, 262144, arena 3072 — coherent, but still not the control's tokens
(diverges at index 2, control resumes `Zhilínski.` where the streaming arm emits
`ilínski.`), which is why the text-coherence reading alone is misleading.

## What your own probe would have picked — and why I think 3072 was my mistake

I picked 3072 by hand. Before writing this up I read
`benchmarks/benchmark_kv_stream.py` at `d7680c6` (my local checkout `diff`s clean
against the raw file at that commit) to see what your tooling would have chosen.

`probe_arena` + `estimate_arena_mib`:

1. start a server at `--kv-stream-arena-mib 1536` (`--probe-arena-mib` default);
   on failure `+256` MiB, up to 8 retries
2. after model init and one warm-up completion, read `nvidia-smi` **free** VRAM
3. `selected = (probe_arena + free) // 32 * 32` — no safety reserve, which
   `benchmarks/README.md` states outright
4. validate on the real point; on failure `-64` MiB, up to 8 times, never below
   `probe_arena_mib`

Because step 3 adds *all* free VRAM, the selection is effectively
`accounted_vram - idle - non_arena_footprint`. I solved the non-arena footprint from
my own measured `peak_vram_mib` (`peak = idle + base + arena`), and four independent
arms agree to within 145 MiB:

| arm | peak MiB | arena | implied non-arena base |
|---|---|---|---|
| ctx262144 arena3072 | 16479 | 3072 | 12212 |
| ctx262144 arena6144 | 19509 | 6144 | 12170 |
| ctx131072 arena3072 | 16582 | 3072 | 12315 |
| ctx131072 arena1024 | 14505 | 1024 | 12286 |

With `nvidia-smi` accounting 24120 MiB used+free and idle 1195 MiB:

| assumed base | free at probe | **probe would select** | backoff floor |
|---|---|---|---|
| 12170 | 9219 | **10752 MiB** | 10240 |
| 12315 | 9074 | **10592 MiB** | 10080 |

So your probe would pick ~10.6–10.8 GiB — about **3.5x** my hand-picked 3072, and
well above the 6144 point that was both fast and token-exact. 3072 sits *below the
probe's backoff floor*; the probe would raise `RuntimeError` before descending that
far. **On your own tooling's terms, 3072 is a number that would never have been
chosen at this context on this card.**

(Arithmetic caveat: I did not run the probe live — the card was busy — this is code
reading plus solving against measured VRAM. It is also a tight fit: by the same
arithmetic 10656 fits and 11104 does not, which is consistent with 12288 failing to
allocate, but the backoff loop is what would settle it in practice.)

## My questions

1. **Is 3072 MiB at 262144 ctx simply outside the supported operating range on a
   24 GiB card?** If it is, the ask is small: 12288 already fails *loudly*
   (`ggml_backend_cuda_phase_arena_new: out of memory`), but 3072 starts fine and
   silently serves fluent-but-divergent output. A guard or warning at partition time
   — "this arena cannot hold an exact partition for this context" — would turn a
   silent wrong answer into a refusal.

2. **If 3072 *is* in range,** then this is a correctness issue against the design
   claim that "every attention layer still processes its complete KV history; only
   physical residency and transfer scheduling change."

3. **What does `benchmark_kv_stream.py`'s probe actually select for a 24 GiB card at
   262144?** My arithmetic above says ~10592–10752 MiB. Does that match what you'd
   expect, or is my reconstruction of `estimate_arena_mib` wrong?

4. **Is there a documented way to compute the minimum workable
   `--kv-stream-arena-mib` for a given context and model**, so "misconfigured" can be
   told apart from "broken" without a sweep?

### What makes me unsure this is pure arena starvation

The 131072/1024 arm thrashes *harder* by copy-busy % (85.1 vs 82.3) with **fewer**
resident pages/layer (130 vs 430), and it is still exactly identical to its control.
So divergence does not track eviction pressure monotonically. That is the single fact
I cannot fit into a clean "you just starved it" story, and it is why I'm asking
rather than filing.

### Caveat I want to state plainly

Divergence proves the two paths **disagree**. It does not by itself prove the
streaming path is the wrong one — I am treating streaming-off as the reference
because it is the unmodified path, not because I have independently verified it at
this context length. A logit-level comparison (KL or top-k overlap at the first
diverging position, rather than argmax token equality) would quantify how large the
disagreement actually is and say whether it is a marginal-tie flip or a real
attention error. I'm happy to run that if it would be useful to you.

## Raw data

`akv-longctx-bench-20260928T015156Z.tgz` in this repo — 234,012 bytes, md5
`d64013912818febc4e2fd9fea6762116`. Contains per-arm server logs, `results.jsonl`,
`results.txt`, `samples.txt`, `gpu.txt`, and the bench driver.
