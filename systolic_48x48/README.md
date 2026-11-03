# systolic_48x48 — Systolic array with hoisted A-strip load (FP32)

An attempt to scale the systolic array to **48×48** for more MACs, plus an
optimization that loads each row-strip of A only once.

> ⚠️ **Reality check:** the target 48×48 array hit **"Congestion Level 7"**
> routing failures during implementation, so the committed kernel is scaled back
> to `SA_DIM = 32`. The directory name records the original intent; the code is
> the version that actually closes timing/routing.

- **Kernel:** `kernel/mmult_systolic.cpp` · top function `mmult_systolic`
- **Precision:** FP32
- **Array:** `SA_DIM = 32` (reduced from 48)

## What's different from `systolic_32x32`

- **Hoisted A load.** A full **row-strip of A** (`SA_DIM` rows × N cols) is loaded
  into URAM *once* at the top of the row loop and reused across every column tile,
  instead of reloading A for each output tile — cutting redundant memory traffic.
- Otherwise it shares the wide-512-bit, transposed-B, URAM-backed systolic
  structure of `systolic_32x32`.

This variant is the practical lesson that **bigger arrays aren't free**: DSP count
scales as `SA_DIM²`, but so does routing pressure, and the U50 fabric could not
place-and-route the 48×48 mesh at target frequency.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult_systolic.xclbin -d 1 -m 256 -n 256 -p 256
```
