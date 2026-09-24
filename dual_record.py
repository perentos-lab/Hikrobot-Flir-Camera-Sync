"""
Record a Hikrobot GigE camera (MVS SDK) and a FLIR/Teledyne Blackfly USB camera
(Spinnaker / PySpin) at the same time.

Output (in C:\\Users\\<you>\\camera_recordings\\<date_time>\\ by default, outside OneDrive):
    hik_000.avi, hik_001.avi, ...     Hikrobot video, one file per segment (default 10 min)
    flir_000.avi, flir_001.avi, ...   Blackfly video, same segmenting
    timestamps.csv   one row per received frame, both cameras, common PC clock,
                     with the video file and frame number inside it

Long recordings:
    python dual_record.py --fps 30 --segment-minutes 10 --jpeg-quality 85
    The status line every 10 s shows GB used, GB/hour and hours of disk left;
    recording stops cleanly when free space drops below --min-free-gb.

Usage:
    python dual_record.py                    # record until Ctrl+C
    python dual_record.py --seconds 60       # fixed duration
    python dual_record.py --fps 30           # force both cameras to 30 fps
    python dual_record.py --no-flir          # test the Hikrobot alone
    python dual_record.py --no-hik           # test the Blackfly alone

Close the MVS app and SpinView before running: the script takes control of both cameras.
Set exposure / gain / pixel format in those apps first; the cameras keep the settings
until power-cycled (or save them to a User Set).

Requires:  Hikrobot MVS (with runtime), Spinnaker SDK + matching PySpin wheel,
           pip install numpy opencv-python
"""
import argparse
import csv
import datetime
import os
import queue
import shutil
import sys
import threading
import time
from ctypes import byref, cast, memset, sizeof, POINTER, c_ubyte

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Common clock: epoch-anchored, but advanced with the high-resolution counter
# ---------------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.epoch0 = time.time_ns()
        self.perf0 = time.perf_counter_ns()

    def now_ns(self):
        return self.epoch0 + (time.perf_counter_ns() - self.perf0)


CLOCK = Clock()

# Default output folder: inside the user profile but NOT in OneDrive-synced Documents
DEFAULT_OUTDIR = os.path.join(os.path.expanduser("~"), "camera_recordings")


def folder_size(path):
    total = 0
    with os.scandir(path) as it:
        for e in it:
            if e.is_file():
                try:
                    total += e.stat().st_size
                except OSError:
                    pass
    return total


# ---------------------------------------------------------------------------
# Video writer running in its own thread so encoding never slows grabbing
# ---------------------------------------------------------------------------
class VideoSink(threading.Thread):
    """Writes frames to a series of video files (segments) of fixed length.

    Segments keep each file a manageable size, let finished files be closed and
    usable while recording continues, and limit the loss if the PC crashes.
    """

    def __init__(self, folder, name, fps, segment_frames, quality):
        super().__init__(daemon=True)
        self.folder, self.name = folder, name
        self.fps = max(fps, 1.0)
        self.segment_frames = segment_frames  # 0 = a single file
        self.quality = quality
        self.q = queue.Queue(maxsize=300)
        self.writer = None
        self.cur_seg = -1
        self.next_idx = 0
        self.skipped = 0
        self.files = []
        self.error = None

    def _loc(self, idx):
        if self.segment_frames:
            return idx // self.segment_frames, idx % self.segment_frames
        return 0, idx

    def filename(self, seg):
        return f"{self.name}_{seg:03d}.avi"

    def submit(self, img):
        """Queue a frame. Returns (video file, frame number inside it), or ("", -1) if the queue was full."""
        try:
            self.q.put_nowait(img)
        except queue.Full:
            self.skipped += 1
            return "", -1
        seg, frame = self._loc(self.next_idx)
        self.next_idx += 1
        return self.filename(seg), frame

    def _open(self, seg, img):
        if self.writer is not None:
            self.writer.release()
        h, w = img.shape[:2]
        path = os.path.join(self.folder, self.filename(seg))
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                                      self.fps, (w, h), isColor=(img.ndim == 3))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        self.writer.set(cv2.VIDEOWRITER_PROP_QUALITY, self.quality)
        self.cur_seg = seg
        self.files.append(path)

    def run(self):
        idx = 0
        try:
            while True:
                img = self.q.get()
                if img is None:
                    break
                seg, _ = self._loc(idx)
                if seg != self.cur_seg:
                    self._open(seg, img)
                self.writer.write(img)
                idx += 1
        except Exception as e:  # keep draining so the grab thread never blocks
            self.error = e
            while self.q.get() is not None:
                pass
        finally:
            if self.writer is not None:
                self.writer.release()

    def close(self):
        self.q.put(None)
        self.join()


# ---------------------------------------------------------------------------
# Hikrobot (MVS SDK)
# ---------------------------------------------------------------------------
def load_hik_sdk():
    mvimport = os.environ.get("MVIMPORT") or (
        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
        if os.name == "nt" else "/opt/MVS/Samples/64/Python/MvImport")
    sys.path.append(mvimport)
    if os.name == "nt":
        runenv = os.environ.get("MVCAM_COMMON_RUNENV") or r"C:\Program Files (x86)\Common Files\MVS\Runtime"
        os.environ["MVCAM_COMMON_RUNENV"] = runenv
        dll_dir = os.path.join(runenv, "Win64_x64" if sys.maxsize > 2**32 else "Win32_i86")
        if not os.path.isfile(os.path.join(dll_dir, "MvCameraControl.dll")):
            raise RuntimeError(f"MvCameraControl.dll not found in {dll_dir}. "
                               "Install/repair MVS or set MVCAM_COMMON_RUNENV.")
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    import MvCameraControl_class as hk
    import PixelType_header as hpt
    return hk, hpt


class HikCamera:
    name = "hik"

    def __init__(self, index=0, fps=None):
        self.hk, self.hpt = hk, hpt = load_hik_sdk()
        if hasattr(hk.MvCamera, "MV_CC_Initialize"):
            hk.MvCamera.MV_CC_Initialize()

        dev_list = hk.MV_CC_DEVICE_INFO_LIST()
        self._check(hk.MvCamera.MV_CC_EnumDevices(hk.MV_GIGE_DEVICE | hk.MV_USB_DEVICE, dev_list), "EnumDevices")
        if dev_list.nDeviceNum <= index:
            raise RuntimeError(f"Hikrobot camera index {index} not found ({dev_list.nDeviceNum} detected)")
        dev_info = cast(dev_list.pDeviceInfo[index], POINTER(hk.MV_CC_DEVICE_INFO)).contents

        self.cam = hk.MvCamera()
        self._check(self.cam.MV_CC_CreateHandle(dev_info), "CreateHandle")
        ret = self.cam.MV_CC_OpenDevice(hk.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"Hikrobot OpenDevice failed: 0x{ret:x} (is the MVS app still connected?)")

        self.is_gige = dev_info.nTLayerType == hk.MV_GIGE_DEVICE
        if self.is_gige:
            pkt = self.cam.MV_CC_GetOptimalPacketSize()
            if int(pkt) > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)

        self.cam.MV_CC_SetEnumValue("TriggerMode", hk.MV_TRIGGER_MODE_OFF)
        if fps:
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self._check(self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps)), "Set frame rate")
        self.cam.MV_CC_SetImageNodeNum(30)  # SDK-side buffers against short stalls

        self.payload = self._get_int("PayloadSize")
        if not self.payload:
            raise RuntimeError("Could not read Hikrobot PayloadSize")
        self.tick_freq = self._get_int("GevTimestampTickFrequency") if self.is_gige else None
        self.buf = (c_ubyte * self.payload)()
        self.info = hk.MV_FRAME_OUT_INFO_EX()
        memset(byref(self.info), 0, sizeof(self.info))
        self.grabbing = False

    # helpers
    @staticmethod
    def _check(ret, what):
        if ret != 0:
            raise RuntimeError(f"Hikrobot {what} failed: 0x{ret:x}")

    def _get_int(self, node):
        hk = self.hk
        if hasattr(self.cam, "MV_CC_GetIntValueEx"):
            v = hk.MVCC_INTVALUE_EX()
            memset(byref(v), 0, sizeof(v))
            ret = self.cam.MV_CC_GetIntValueEx(node, v)
        else:
            v = hk.MVCC_INTVALUE()
            memset(byref(v), 0, sizeof(v))
            ret = self.cam.MV_CC_GetIntValue(node, v)
        return v.nCurValue if ret == 0 else None

    def frame_rate(self):
        v = self.hk.MVCC_FLOATVALUE()
        memset(byref(v), 0, sizeof(v))
        for node in ("ResultingFrameRate", "AcquisitionFrameRate"):
            if self.cam.MV_CC_GetFloatValue(node, v) == 0 and v.fCurValue > 0:
                return v.fCurValue
        return 30.0

    def _to_image(self, info):
        hpt = self.hpt
        w, h, pt = info.nWidth, info.nHeight, info.enPixelType
        raw = np.frombuffer(self.buf, dtype=np.uint8, count=info.nFrameLen)
        if pt == hpt.PixelType_Gvsp_Mono8:
            return raw[: w * h].reshape(h, w).copy()
        bayer = {  # GenICam -> OpenCV naming is shifted by one row
            hpt.PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
            hpt.PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
            hpt.PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
            hpt.PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
        }
        if pt in bayer:
            return cv2.cvtColor(raw[: w * h].reshape(h, w), bayer[pt])
        if pt == hpt.PixelType_Gvsp_RGB8_Packed:
            return cv2.cvtColor(raw[: w * h * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if pt == hpt.PixelType_Gvsp_BGR8_Packed:
            return raw[: w * h * 3].reshape(h, w, 3).copy()
        raise RuntimeError(f"Hikrobot pixel format 0x{pt:x} not supported for video; "
                           "set Mono8 / Bayer 8-bit / RGB8 in MVS")

    # acquisition
    def start(self):
        self._check(self.cam.MV_CC_StartGrabbing(), "StartGrabbing")
        self.grabbing = True

    def grab(self, timeout_ms=1000):
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.buf), self.payload, self.info, timeout_ms)
        pc_ns = CLOCK.now_ns()
        if ret != 0:
            return None
        info = self.info
        ticks = (info.nDevTimeStampHigh << 32) | info.nDevTimeStampLow
        incomplete = info.nLostPacket > 0
        img = None if incomplete else self._to_image(info)
        meta = {
            "cam_frame_id": info.nFrameNum,
            "dev_ts_raw": ticks,
            "dev_ts_s": ticks / self.tick_freq if self.tick_freq else "",
            "pc_time_ns": pc_ns,
            "incomplete": int(incomplete),
        }
        return img, meta

    def stop(self):
        if self.grabbing:
            self.cam.MV_CC_StopGrabbing()
            self.grabbing = False

    def close(self):
        self.stop()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        if hasattr(self.hk.MvCamera, "MV_CC_Finalize"):
            self.hk.MvCamera.MV_CC_Finalize()


# ---------------------------------------------------------------------------
# FLIR / Teledyne Blackfly (Spinnaker / PySpin)
# ---------------------------------------------------------------------------
class FlirCamera:
    name = "flir"

    def __init__(self, index=0, fps=None, serial=None):
        import PySpin
        self.ps = PySpin
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        try:
            self.cam = self._select(index, serial)
        except Exception:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise
        self.cam.Init()

        # Stream buffers: keep every frame in order, with room for short stalls
        s_map = self.cam.GetTLStreamNodeMap()
        self._set_enum(s_map, "StreamBufferHandlingMode", "OldestFirst")
        self._set_enum(s_map, "StreamBufferCountMode", "Manual")
        cnt = PySpin.CIntegerPtr(s_map.GetNode("StreamBufferCountManual"))
        if PySpin.IsWritable(cnt):
            cnt.SetValue(min(100, cnt.GetMax()))

        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        try:
            self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        except PySpin.SpinnakerException:
            pass
        if fps:
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.cam.AcquisitionFrameRate.SetValue(float(fps))

        self.processor = PySpin.ImageProcessor()
        self.processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        self.grabbing = False

    def _select(self, index, serial):
        """Pick the FLIR camera.

        Spinnaker also enumerates GigE cameras from other vendors (your Hikrobot
        shows up in its list), so never trust a bare list index: filter to real
        FLIR/Teledyne devices first, or match an exact serial number.
        """
        ps = self.ps
        found = []
        for i in range(self.cam_list.GetSize()):
            c = self.cam_list.GetByIndex(i)
            try:
                nm = c.GetTLDeviceNodeMap()

                def s(name):
                    node = ps.CStringPtr(nm.GetNode(name))
                    return node.GetValue() if ps.IsReadable(node) else ""

                found.append((i, s("DeviceVendorName"), s("DeviceModelName"), s("DeviceSerialNumber")))
            except ps.SpinnakerException:
                found.append((i, "", "", ""))
            finally:
                del c

        if not found:
            raise RuntimeError("Spinnaker sees no cameras at all")

        listing = "\n".join(f"    [{i}] {v} {m} serial {sn}" for i, v, m, sn in found)

        if serial:
            hits = [f for f in found if f[3] == str(serial)]
            if not hits:
                raise RuntimeError(f"No Spinnaker camera with serial {serial}. Saw:\n{listing}")
            pick = hits[0]
        else:
            flir = [f for f in found
                    if "flir" in f[1].lower() or "point grey" in f[1].lower()
                    or "teledyne" in f[1].lower() or "blackfly" in f[2].lower()]
            if not flir:
                raise RuntimeError(f"No FLIR/Teledyne camera among the Spinnaker devices. Saw:\n{listing}")
            if index >= len(flir):
                raise RuntimeError(f"FLIR index {index} out of range ({len(flir)} FLIR cameras). Saw:\n{listing}")
            pick = flir[index]

        print(f"  using FLIR {pick[2]} serial {pick[3]} (Spinnaker index {pick[0]})")
        return self.cam_list.GetByIndex(pick[0])

    def _set_enum(self, nodemap, node, entry):
        ps = self.ps
        n = ps.CEnumerationPtr(nodemap.GetNode(node))
        if ps.IsWritable(n):
            e = n.GetEntryByName(entry)
            if ps.IsReadable(e):
                n.SetIntValue(e.GetValue())

    def frame_rate(self):
        try:
            return float(self.cam.AcquisitionResultingFrameRate.GetValue())
        except self.ps.SpinnakerException:
            return 30.0

    def start(self):
        self.cam.BeginAcquisition()
        self.grabbing = True

    def grab(self, timeout_ms=1000):
        ps = self.ps
        try:
            image = self.cam.GetNextImage(timeout_ms)
        except ps.SpinnakerException:
            return None  # timeout
        pc_ns = CLOCK.now_ns()
        try:
            incomplete = image.IsIncomplete()
            ts_ns = image.GetTimeStamp()  # device clock, nanoseconds
            meta = {
                "cam_frame_id": image.GetFrameID(),
                "dev_ts_raw": ts_ns,
                "dev_ts_s": ts_ns / 1e9,
                "pc_time_ns": pc_ns,
                "incomplete": int(incomplete),
            }
            img = None
            if not incomplete:
                if image.GetPixelFormat() == ps.PixelFormat_Mono8:
                    img = image.GetNDArray().copy()
                else:
                    img = self.processor.Convert(image, ps.PixelFormat_BGR8).GetNDArray().copy()
        finally:
            image.Release()
        return img, meta

    def stop(self):
        if self.grabbing:
            self.cam.EndAcquisition()
            self.grabbing = False

    def close(self):
        self.stop()
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
CSV_HEADER = ["camera", "frame_idx", "cam_frame_id", "dev_ts_raw", "dev_ts_s",
              "pc_time_ns", "incomplete", "video_file", "video_frame"]


def camera_worker(cam, sink, rows_q, barrier, stop_event, counts, errors):
    try:
        barrier.wait()          # both cameras start at (almost) the same instant
        cam.start()
        n = 0
        while not stop_event.is_set():
            r = cam.grab(1000)
            if r is None:
                continue
            img, m = r
            n += 1
            vfile, vframe = sink.submit(img) if img is not None else ("", -1)
            rows_q.put([cam.name, n, m["cam_frame_id"], m["dev_ts_raw"], m["dev_ts_s"],
                        m["pc_time_ns"], m["incomplete"], vfile, vframe])
            counts[cam.name] = n
    except Exception as e:
        errors.append(f"{cam.name}: {e}")
        stop_event.set()
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        try:
            cam.stop()
        except Exception:
            pass


def csv_worker(path, rows_q):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        n = 0
        while True:
            row = rows_q.get()
            if row is None:
                break
            w.writerow(row)
            n += 1
            if n % 200 == 0:
                f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="duration (0 = until Ctrl+C)")
    ap.add_argument("--fps", type=float, default=None, help="set this frame rate on both cameras")
    ap.add_argument("--hik-index", type=int, default=0)
    ap.add_argument("--flir-index", type=int, default=0,
                    help="index among the FLIR cameras only (non-FLIR devices are ignored)")
    ap.add_argument("--flir-serial", default=None,
                    help="select the FLIR camera by exact serial number, e.g. 18285938")
    ap.add_argument("--no-hik", action="store_true")
    ap.add_argument("--no-flir", action="store_true")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help=f"where recordings go (default {DEFAULT_OUTDIR}; keep it OUT of OneDrive)")
    ap.add_argument("--segment-minutes", type=float, default=10,
                    help="start a new video file every N minutes (0 = one file per camera)")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="MJPEG quality 1-100; lower = smaller files (default 90)")
    ap.add_argument("--min-free-gb", type=float, default=10,
                    help="stop recording cleanly when free disk space drops below this")
    args = ap.parse_args()

    out = os.path.abspath(os.path.join(args.outdir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    if "onedrive" in out.lower():
        print("WARNING: output folder is inside OneDrive. OneDrive will try to upload the videos\n"
              "         while they are being written, which slows the PC and hangs Explorer.\n"
              f"         Use --outdir with a folder outside OneDrive (default: {DEFAULT_OUTDIR}).")
    os.makedirs(out, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    print(f"Output: {out}  ({free_gb:.0f} GB free)")
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on the output drive")

    cams = []
    try:
        if not args.no_hik:
            print("Opening Hikrobot...")
            cams.append(HikCamera(args.hik_index, args.fps))
        if not args.no_flir:
            print("Opening FLIR Blackfly...")
            cams.append(FlirCamera(args.flir_index, args.fps, args.flir_serial))
        if not cams:
            raise RuntimeError("Both cameras disabled")
    except Exception:
        for c in cams:
            c.close()
        raise

    sinks = {}
    for c in cams:
        fr = c.frame_rate()
        seg_frames = int(round(fr * args.segment_minutes * 60)) if args.segment_minutes > 0 else 0
        print(f"  {c.name}: {fr:.2f} fps" + (f", new file every {seg_frames} frames" if seg_frames else ""))
        sinks[c.name] = VideoSink(out, c.name, fr, seg_frames, args.jpeg_quality)
        sinks[c.name].start()

    rows_q = queue.Queue()
    csv_path = os.path.join(out, "timestamps.csv")
    csv_thread = threading.Thread(target=csv_worker, args=(csv_path, rows_q), daemon=True)
    csv_thread.start()

    stop_event = threading.Event()
    barrier = threading.Barrier(len(cams))
    counts = {c.name: 0 for c in cams}
    errors = []
    threads = [threading.Thread(target=camera_worker,
                                args=(c, sinks[c.name], rows_q, barrier, stop_event, counts, errors),
                                daemon=True) for c in cams]
    for t in threads:
        t.start()

    print(f"Recording to {out}  (Ctrl+C to stop)")
    t0 = time.time()
    last_status = t0
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
            if args.seconds and time.time() - t0 >= args.seconds:
                break
            if time.time() - last_status >= 10:
                last_status = time.time()
                el = last_status - t0
                used_gb = folder_size(out) / 1e9
                free_gb = shutil.disk_usage(out).free / 1e9
                per_hour = used_gb / el * 3600
                hours_left = (free_gb - args.min_free_gb) / per_hour if per_hour > 0 else float("inf")
                cams_txt = "   ".join(f"{k}: {v} fr ({v / el:.1f} fps, backlog {sinks[k].q.qsize()})"
                                      for k, v in counts.items())
                print(f"  [{datetime.timedelta(seconds=int(el))}] {cams_txt}   "
                      f"{used_gb:.1f} GB (~{per_hour:.0f} GB/h, ~{hours_left:.1f} h of space left)")
                if free_gb < args.min_free_gb:
                    print(f"Free space below {args.min_free_gb} GB -> stopping.")
                    break
    except KeyboardInterrupt:
        print("Stopping...")
    stop_event.set()

    for t in threads:
        t.join(timeout=5)
    for s in sinks.values():
        s.close()
    rows_q.put(None)
    csv_thread.join()
    for c in cams:
        try:
            c.close()
        except Exception as e:
            print(f"Warning closing {c.name}: {e}")

    print(f"\nSaved to {out}")
    for name, n in counts.items():
        s = sinks[name]
        extra = f", {s.skipped} not written to video (encoder too slow)" if s.skipped else ""
        if s.error:
            extra += f", VIDEO ERROR: {s.error}"
        print(f"  {name}: {n} frames in {len(s.files)} video file(s){extra}")
    for e in errors:
        print(f"ERROR {e}")


if __name__ == "__main__":
    main()