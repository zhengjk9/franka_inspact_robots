"""RealSense camera reader for the Franka embodiment.

The stock ``inspect-robots-franka`` adapter reads cameras with OpenCV over
V4L2, which drops frames / times out when several D435 units share one USB
bus while the arm is under real-time control (observed ``select() timeout``
in production). This module reuses the ``pyrealsense2`` driver that the
openpi rig (``examples/franka_real``) already runs stably: it picks cameras
by serial number and streams RGB plus aligned metric depth at a fixed rate.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_CAMERA_SERIALS: dict[str, str] = {
    # keys are the inspect_robots_franka camera names; values are RealSense
    # serial numbers as enumerated by pyrealsense2 on this host. The rig runs
    # two exterior views and one wrist camera; exterior_cam_2 is the second
    # (side) exterior camera.
    "exterior_cam": "317622073233",
    "wrist_cam": "242322072531",
    "exterior_cam_2": "250122070653",
}
_WIDTH = 640
_HEIGHT = 480
_FPS = 15
# Manual white balance in Kelvin. The D435 auto-white-balance can freeze at a
# value that renders greenish under these office/LED lights; locking it at this
# neutral point (closest to R==G==B in testing) removes the green cast.
_WHITE_BALANCE = 5000


def _pick(mapping: dict[str, str]) -> dict[str, str]:
    """Return the RS serials for every camera name in the mapping, resolving by name."""
    import pyrealsense2 as rs

    ctx = rs.context()
    by_serial = {d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()}
    resolved: dict[str, str] = {}
    for name, serial in mapping.items():
        if serial not in by_serial:
            raise RuntimeError(f"RealSense serial {serial!r} for {name!r} not found")
        resolved[name] = serial
    if not resolved:
        raise RuntimeError("no cameras configured")
    return resolved


def make_rs_camera_reader(
    *,
    serials: dict[str, str] | None = None,
    width: int = _WIDTH,
    height: int = _HEIGHT,
    fps: int = _FPS,
) -> object:
    """Return an RGB reader with aligned metric depth in its ``depths`` attribute.

    RGB arrays are uint8 HWC; depth arrays are float32 HW in metres. Zero
    depth denotes a missing measurement. Each call replaces the depth snapshot.
    """
    return _RealSenseReader(dict(_CAMERA_SERIALS if serials is None else serials),
                            width, height, fps)


class _RealSenseReader:
    def __init__(self, mapping, width, height, fps):
        self.mapping = mapping
        self.width, self.height, self.fps = width, height, fps
        self.pipelines = []
        self.depths: dict[str, npt.NDArray[np.float32]] = {}

    def close(self):
        """Release every started stream, including after partial startup failures."""
        import contextlib

        for _, pipe, _, _ in self.pipelines:
            with contextlib.suppress(Exception):
                pipe.stop()
        self.pipelines.clear()
        self.depths = {}

    def _start(self):
        import pyrealsense2 as rs

        try:
            for name, serial in _pick(self.mapping).items():
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, self.width, self.height,
                                  rs.format.bgr8, self.fps)
                cfg.enable_stream(rs.stream.depth, self.width, self.height,
                                  rs.format.z16, self.fps)
                pipe = rs.pipeline()
                profile = pipe.start(cfg)
                # Register immediately so later sensor setup failures stop this stream.
                self.pipelines.append((name, pipe, None, None))
                scale = profile.get_device().first_depth_sensor().get_depth_scale()
                if not np.isfinite(scale) or scale <= 0:
                    raise RuntimeError(f"invalid depth scale for {name}: {scale}")
                self.pipelines[-1] = (name, pipe, rs.align(rs.stream.color), scale)
                for sensor in profile.get_device().query_sensors():
                    if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                        sensor.set_option(rs.option.enable_auto_white_balance, 0)
                        sensor.set_option(rs.option.white_balance, _WHITE_BALANCE)
            for _ in range(max(1, int(self.fps))):
                for _, pipe, _, _ in self.pipelines:
                    pipe.wait_for_frames(5000)
        except BaseException:
            self.close()
            raise

    def __call__(self) -> dict[str, npt.NDArray[np.uint8]]:
        if not self.pipelines:
            self._start()
        frames, depths = {}, {}
        self.depths = {}
        try:
            for name, pipe, align, scale in self.pipelines:
                aligned = align.process(pipe.wait_for_frames(5000))
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color or not depth:
                    raise RuntimeError(f"missing aligned RGB/depth frame for {name}")
                bgr = np.asanyarray(color.get_data())
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                metric = np.asarray(depth.get_data(), dtype=np.float32) * scale
                if metric.shape != rgb.shape[:2]:
                    raise RuntimeError(f"RGB/depth shape mismatch for {name}")
                frames[name] = rgb
                depths[name] = metric
        except BaseException:
            self.close()
            raise
        self.depths = depths
        return frames
