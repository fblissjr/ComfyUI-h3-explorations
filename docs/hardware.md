# The box, and what actually bounds it

Last updated: 2026-08-17.

Every measured number in this repo was taken on one machine. This file is what
about that machine can change a result, and how to tell whether it has. It does
not carry the machine's current values — `bench/hwinfo.py` prints those, because
driver versions, power limits and clock states drift and CLAUDE.md's rule is
that a drifting number does not belong in prose.

```
python bench/hwinfo.py
```

Run it before quoting a timing number and after any host change. It is not a
check; it asserts nothing and only reports.

## The short version

| if you are about to | read |
|---|---|
| quote or compare a render time | [Host power state](#host-power-state-invalidates-timing) — first, and it is the one people skip |
| reason about why a knob did or did not help | [What bounds this workload](#what-bounds-this-workload) |
| blame the host for a slow render | [What is not the bottleneck](#what-is-not-the-bottleneck) |
| change a GPU-level setting | [Host settings and their blast radius](#host-settings-and-their-blast-radius) |

## Host power state invalidates timing

A GPU board power limit changes render times, is set outside the repo, and
survives reboots the moment somebody installs a systemd unit for it. A render at
a changed limit produces a perfectly ordinary-looking s/it that is not
comparable to any number in [`docs/bench_plan.md`](bench_plan.md) or
[`docs/SOLATTN.md`](SOLATTN.md). Every timing figure in those two files was
taken at the card's stock limit.

**How well it is recorded depends entirely on which artifact you are holding**,
and the three cases are not the same defect:

| artifact | host power state | so what |
|---|---|---|
| capture manifest | `provenance.gpu_power_limit_watts` **exists**, and the one manifest on disk populates it | and as of 2026-08-17 it is required and asserted, so a manifest that omits it fails. It is also **read** now rather than typed in -- the generator wrote a literal `450.0` until the same day |
| render stamp | **no power, driver or GPU field at all** | a stamped render cannot be placed on a power state even in principle |
| bench run | **persisted per row since 2026-08-18** by `bench/run_graph_arms.py` (`substrate.power_limit`, `power_limit_is_stock`, driver, package versions, git commit); `bench/bench_e2e_h3.py` still persists nothing | the runner that carries the day's timings now records its own conditions; the older harness does not |

So the fix is not one fix. For captures it is an assertion on a field that is
already there; for stamps it is a field that does not exist; for bench runs it
is deciding a bench run should emit a record at all.

`bench/hwinfo.py` prints the limit and flags it when it differs from the
default. That is a report a person chooses to run, not a guard — and no check
can see the power state of a run whose number is already written down.

**A correction, kept because it is instructive.** This section originally said
the power limit was "invisible everywhere this repo looks", naming the capture
manifest among the places it was absent. That was wrong: the field existed, and
was found by a peer session the same day. The error came from reading the
schema's `required` list and inferring the property did not exist — an inference
stated as a fact about the file, which is the failure
[`docs/evidence.md`](evidence.md) exists to catch. The distinction matters
practically, not just for the record: a field that exists and is unasserted is
fixed by an assertion, and a field that is absent is fixed by adding one. Only
the second was the work originally proposed here.

This remains a live example of CLAUDE.md's rule about requirements without
controls: the requirement is "compare timings only at equal power state", and
the control is a script somebody has to choose to run. It is **not** enforced.

Because the limit is a host setting rather than a repo setting, the same caution
applies to anything else that changes the machine's steady state under load:
ambient temperature, case airflow, a concurrent job on the same card, and the
persistence-mode setting that decides whether the driver was already resident
when the run started.

## What bounds this workload

The honest answer is that it depends on the phase, and the parts have never been
separated on this box. What follows is the reasoning frame, not a verdict.

### The model does not fit

Measured 2026-08-17: the DiT, text encoder and VAE of a reference render exceed
this card's memory together, and the DiT alone exceeds what stays resident once
activations for a full-length packed sequence are accounted for. So ComfyUI
streams blocks rather than holding everything resident. **This paragraph used
to conclude that PCIe therefore carries continuous traffic through the whole
sampling loop. Measured 2026-08-18 and it does not, at the one configuration
measured:** `bench/results/2026-08-18_phase0_instrument.json` recorded ~20 MB/s
of PCIe receive during sampling at 1024x768 with three references, with power
pegged at the 450 W limit and core clocks pulled ~24% under their maximum.
Sampling at that configuration is compute/power-bound, not host-bound; the
streaming happens between phases. Heavier configurations are untested.
`hwinfo.py` prints the resident figure; `nvidia-smi dmon -s pumt` shows the
traffic. Since 2026-08-25 `bench/instrument_render_occupancy.py` takes the
same reading per node of one render at 100 ms and states, with its
thresholds, whether the sampler window was launch-bound or compute-bound;
that verdict is the test for whether CUDA-graph replay could buy anything on
a graph, and `docs/research/sglang_comparison.md` records why it is not
expected to at video canvases.

Anything reasoning from "the model is resident, so the sampling loop touches no
host interface" is wrong here, and that assumption is easy to make because it
holds on smaller graphs.

### Attention dominates, but "bandwidth-bound" is three different claims

Most sampling time is inside attention. That does not by itself say which
resource is binding, and the distinction matters because the three candidates
respond to a core clock change in completely different ways:

| bound by | clock domain | responds to a core power limit |
|---|---|---|
| PCIe / host staging | link, independent of GPU core | no |
| GDDR6X device memory | memory controller, independent of core V/F | no |
| L2 and shared memory | **GPU core clock** | **yes, roughly in step** |
| tensor-core issue rate | **GPU core clock** | **yes, roughly in step** |

Sol-Attn's block sparsity pulls in two directions at once. It cuts arithmetic
faster than it cuts key and value read traffic, and gathered block reads
coalesce worse than dense streaming, which leans toward bandwidth. But it
gathers into L2 and shared memory, which are on the core clock domain. So
"sparse attention is bandwidth-bound, therefore core clock is free" does not
follow, and it is worth stating because it is the intuitive move.

`docs/SOLATTN.md` owns Sol-Attn's measured arms and its shared-memory
footprints. `docs/open_experiments.md` owns the tensor-core issue rates, which
`bench/mma_rate.cu` reproduces.

### What would settle it

Nsight Compute on a single Sol-Attn kernel, or a paired render at two power
limits on one graph and seed. ~~Neither has been run.~~ **The power pair was
run on 2026-08-20** (`bench/results/2026-08-20_power_limit_pair_verdict.json`):
on a 4-step 768p turbo render the 330 W cap costs 5.8% of sampler time against
a 12.5% core-clock delta, power pegged at both limits. Partly core-clock-bound;
a core clock change is not free here, and it is not proportional either. The
`ncu` half is still unrun, and the 16-step all-refs workload is not measured.

## What is not the bottleneck

Recorded because each of these has been proposed as an explanation and does not
survive measurement.

**PCIe link width.** The GPU negotiates a narrower link than it is capable of on
this box. Sustained traffic during sampling sits far enough below what the
narrow link already provides that widening it would change stage swap and
initial load times, not s/it. `hwinfo.py` flags any device linked below its
capability, and sampling traffic is visible in `nvidia-smi dmon -s pumt`.

The width is worth chasing anyway, but for the load-time win and because the
cause is unexplained rather than known-by-design. Seating and firmware both
produce a narrowed link and neither announces itself; storage devices on
separate root ports do not, which is a plausible-sounding cause that the
topology in `hwinfo.py`'s output refutes.

**Host RAM and swap.** Ample, with a large page cache and no swap pressure. The
page cache is why a repeat stage swap is much cheaper than the first — a second
load of the same stage comes from RAM rather than storage.

**CPU.** Not a factor during sampling. Text and reference encoding run on the
GPU, not on a CPU core, so a claim that host single-thread speed gates prompt
processing is a mechanism error rather than a measurement.

**Temp directory on a network share.** This *was* real, and is fixed: pointing
ComfyUI's temp path at local storage removed intermediate-write stalls on frame
dumps and previews. Left here so it is not rediscovered as an open problem.

## Host settings and their blast radius

| setting | reversible | changes numerics | notes |
|---|---|---|---|
| board power limit (`nvidia-smi -pl`) | yes, instantly | no | invalidates timing comparisons — see above |
| persistence mode (`nvidia-smi -pm`) | yes | no | affects first-run latency only |
| core clock lock (`nvidia-smi -lgc`) | yes | no | a power limit already takes precedence over a clock floor, so the floor is inert under one |
| memory clock lock (`nvidia-smi -lmc`) | yes | **potentially** | see below |

**Memory clock locking deserves its own warning.** The driver deliberately runs
device memory below its graphics-state ceiling while in the compute state.
Overriding that is a memory overclock relative to the compute default, and it is
easy to mistake for restoring stock because the two numbers are close and the
higher one is labelled "max".

GDDR6X link errors on this generation surface as retry and replay rather than as
a crash. The symptom is a slower run, or wrong bits — and this repo writes tensor
checksums into the capture manifest, so a silent flip breaks byte-exact
reproducibility and presents as a capture that will not validate. Do not lock
the memory clock, and especially do not bake it into a boot service, where the
next unexplained checksum mismatch will not look like it came from a setting
made months earlier.

## Reproducing anything here

| what | how |
|---|---|
| host, GPU, power state, PCIe topology | `python bench/hwinfo.py` |
| power, utilisation and PCIe traffic under load | `nvidia-smi dmon -s pumt -c 12 -d 1` |
| tensor-core issue rates | `bench/mma_rate.cu`, owned by `docs/open_experiments.md` |
| whether a render is comparable to a recorded one | `hwinfo.py`, power limit line |

Sample anything PCIe-related **under load**. The link downtrains at idle, so the
speed reads far below what it negotiates once work arrives; width is the half
that stays honest when the card is asleep.

## Related

- [`docs/comfy_notes.md`](comfy_notes.md) — running ComfyUI, restarting it
  correctly, and the settings inside the launcher that are not this file's
  business
- [`docs/bench_plan.md`](bench_plan.md) — the runs whose timings the power-state
  caveat applies to
- [`docs/evidence.md`](evidence.md) — its Environment section is the software
  half of "state that is not in git"; this file is the host half
- [`docs/SOLATTN.md`](SOLATTN.md) — Sol-Attn's own measured numbers and its
  do-not-rely-on table
- [`docs/open_experiments.md`](open_experiments.md) — what is deliberately
  unmeasured, including the bound-separation question above
- [`docs/capture_manifest_schema.md`](capture_manifest_schema.md) — what a
  capture records, which is where the power-state gap shows up as a missing
  field
- [`bench/hwinfo.py`](../bench/hwinfo.py) — the values this file deliberately
  does not carry
