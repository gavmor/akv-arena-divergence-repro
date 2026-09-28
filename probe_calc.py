"""Derive what benchmark_kv_stream.py's arena probe would select at ctx=262144
on the 24 GiB RTX 3090, using measured VRAM from Concourse build 1000936.

Probe algorithm (benchmark_kv_stream.py d7680c6, probe_arena + estimate_arena_mib):
  1. start server with --kv-stream-arena-mib 1536 (--probe-arena-mib default)
  2. if it cannot start, +256 MiB and retry (up to 8 times)
  3. read nvidia-smi free VRAM after model init + warm-up completion
  4. selected = (1536 + free) // 32 * 32          <- no safety reserve at all
  5. validate by running the real point; on failure -64 MiB, up to 8 times
"""

PROBE_ARENA = 1536      # --probe-arena-mib default
STEP = 32               # --arena-step-mib default
BACKOFF = 64            # --arena-backoff-mib default
RETRIES = 8             # --arena-retries default

# nvidia-smi on this card accounts used+free = 24120 MiB (total field says 24576)
ACCOUNTED = 24120
IDLE_USED = 1195        # build 1000936 idle_vram_mib

# non-arena VRAM footprint, solved from build 1000936 peak_vram_mib:
#   peak = idle + non_arena_base + arena
MEASURED = [
    ("ctx262144 arena3072", 16479, 3072),
    ("ctx262144 arena6144", 19509, 6144),
    ("ctx131072 arena3072", 16582, 3072),
    ("ctx131072 arena1024", 14505, 1024),
]
bases = []
for label, peak, arena in MEASURED:
    base = peak - IDLE_USED - arena
    bases.append(base)
    print(f"{label:24s} peak={peak:6d}  arena={arena:6d}  -> non-arena base {base}")

lo, hi = min(bases), max(bases)
print(f"\nnon-arena base band: {lo}-{hi} MiB (model weights + CUDA context)")

# the OOM arm never reached steady state; its peak is the pre-arena footprint
OOM_ARM_PEAK = 13130
print(f"arena=12288 OOM arm peaked at {OOM_ARM_PEAK} MiB "
      f"(= idle + {OOM_ARM_PEAK - IDLE_USED} MiB pre-arena footprint)")

print("\n--- what the probe would select ---")
for base in (lo, hi):
    used_at_probe = IDLE_USED + base + PROBE_ARENA
    free = ACCOUNTED - used_at_probe
    selected = (PROBE_ARENA + free) // STEP * STEP
    floor = selected - RETRIES * BACKOFF
    print(f"base={base}: used@probe={used_at_probe}  free={free}  "
          f"SELECTED={selected} MiB  (backoff floor {floor})")

print("\n--- sanity: would the selection even fit? ---")
pre_arena = OOM_ARM_PEAK - IDLE_USED
for cand in (3072, 6144, 10400, 10656, 11104, 12288):
    total = IDLE_USED + pre_arena + cand
    print(f"arena={cand:6d}: idle+pre-arena+arena = {total:6d} MiB "
          f"vs {ACCOUNTED} accounted -> {'FITS' if total <= ACCOUNTED else 'OOM'}")
