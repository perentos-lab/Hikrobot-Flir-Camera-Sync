"""
Analyze a dual_record.py recording and pair frames across the two cameras.

Usage:
    python analyze_sync.py recordings/20260924_103000

Prints per-camera stats (fps, jitter, dropped frames, clock drift) and writes
pairs.csv: for every Hikrobot frame, the nearest Blackfly frame in time,
with the video file + frame number for each camera and the time offset between them.
"""
import os
import sys

import numpy as np
import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "."
df = pd.read_csv(os.path.join(folder, "timestamps.csv"))
df["pc_s"] = df["pc_time_ns"] / 1e9
if "video_file" not in df.columns:  # recordings made before segmenting existed
    df["video_file"] = df["camera"] + ".avi"
df["video_file"] = df["video_file"].fillna("")

cams = {}
for name, g in df.groupby("camera"):
    g = g.sort_values("frame_idx").copy()
    n = len(g)

    gaps = g["cam_frame_id"].diff().dropna()
    dropped = int((gaps - 1).clip(lower=0).sum())

    # Aligned time: map the (steady) device clock onto the common PC clock.
    # A linear fit removes both the offset and the drift between the two clocks.
    dev = pd.to_numeric(g["dev_ts_s"], errors="coerce")
    if dev.notna().all() and n > 10:
        a, b = np.polyfit(dev, g["pc_s"], 1)
        g["t"] = a * dev + b
        resid_ms = (g["pc_s"] - g["t"]).std() * 1000
        drift = f"{(a - 1) * 1e6:+.1f} ppm (PC arrival jitter {resid_ms:.2f} ms)"
    else:
        g["t"] = g["pc_s"]
        drift = "n/a (no device time in seconds) - using PC arrival time"

    dt = g["t"].diff().dropna() * 1000
    print(f"[{name}] frames {n}, dropped {dropped}, incomplete {int(g['incomplete'].sum())}, "
          f"not in video {(g['video_frame'] < 0).sum()}")
    print(f"        fps {1000 / dt.mean():.2f}, interval {dt.mean():.3f} ± {dt.std():.3f} ms")
    print(f"        clock drift vs PC: {drift}")
    cams[name] = g

if {"hik", "flir"} <= cams.keys():
    h = cams["hik"][["t", "frame_idx", "cam_frame_id", "video_file", "video_frame"]].rename(
        columns=lambda c: "hik_" + c if c != "t" else "t")
    f = cams["flir"][["t", "frame_idx", "cam_frame_id", "video_file", "video_frame"]].rename(
        columns=lambda c: "flir_" + c if c != "t" else "flir_t")
    f["t"] = f["flir_t"]

    period = max(cams["hik"]["t"].diff().median(), cams["flir"]["t"].diff().median())
    pairs = pd.merge_asof(h.sort_values("t"), f.sort_values("t"), on="t",
                          direction="nearest", tolerance=period / 2)
    pairs["offset_ms"] = (pairs["flir_t"] - pairs["t"]) * 1000
    pairs = pairs.rename(columns={"t": "hik_t"})
    pairs["time_s"] = pairs["hik_t"] - pairs["hik_t"].iloc[0]
    pairs = pairs[["time_s", "hik_video_file", "hik_video_frame", "flir_video_file", "flir_video_frame",
                   "offset_ms", "hik_frame_idx", "flir_frame_idx", "hik_cam_frame_id", "flir_cam_frame_id"]]
    out = os.path.join(folder, "pairs.csv")
    pairs.to_csv(out, index=False)

    matched = pairs["flir_frame_idx"].notna()
    print(f"\nPaired {matched.sum()} / {len(pairs)} hik frames with a flir frame "
          f"(within ±{period * 500:.1f} ms)")
    print(f"Offset flir - hik: mean {pairs.loc[matched, 'offset_ms'].mean():.2f} ms, "
          f"max |{pairs.loc[matched, 'offset_ms'].abs().max():.2f}| ms")
    print(f"Wrote {out}")