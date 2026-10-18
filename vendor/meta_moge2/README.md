# vendor/meta_moge2: HyDen MoGe-2 metric-point loader and geometry

last updated: 2026-10-17

This folder holds the PyTorch loader and geometry routines for Meta's HyDen MoGe-2 metric-point model,
adapted from Meta's code in
[facebookresearch/metadepth](https://github.com/facebookresearch/metadepth) at commit
`810b77c3e56712813a0de42a130cfbe6f5b19b90` (`mogev2/moge_v2.py`, `mogev2/modules.py`, `mogev2/utils.py`,
`cnn/encoder.py`, and `example_code/hyden_load_and_inference.py`).

These files are under Meta's license in [`LICENSE`](LICENSE).

## What is here

| path | whose | what |
|---|---|---|
| `LICENSE` | Meta's | upstream's `LICENSE`, byte for byte as upstream ships it |
| `hyden_moge2.py` | Meta's, adapted | the HyDen MoGe-2 metric-point model (`HyDenMoGe`), focal/shift and intrinsics recovery, and raw `[0, 1]` `float32` RGB input guard |
| `README.md` | ours | |

## What holds it

[`bench/check_measure_moge2_depth.py`](../../bench/check_measure_moge2_depth.py) holds [`hyden_moge2.py`](hyden_moge2.py)
and [`LICENSE`](LICENSE): the license file to its recorded SHA-256 and to the reference checkout's file where present;
the model's 953 `state_dict` keys (`738` parameters and `215` buffers) to the checkpoint on disk; the focal, shift, and
intrinsics recovery on a synthetic 3D point cloud; and the raw `[0, 1]` `float32` RGB input guard.

## Who reads it

- [`bench/measure_moge2_depth.py`](../../bench/measure_moge2_depth.py) imports `DEFAULT_CHECKPOINT`, `HyDenMoGe`,
  `fov_deg_from_intrinsics`, and `load_hyden_moge2` from [`hyden_moge2.py`](hyden_moge2.py) to measure per-frame metric
  depth and camera intrinsics on video frames.
- [`bench/check_measure_moge2_depth.py`](../../bench/check_measure_moge2_depth.py) imports from [`hyden_moge2.py`](hyden_moge2.py)
  to test the input guard, geometry recovery, and checkpoint schema on CPU.
