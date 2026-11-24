"""Evaluate the gpt-6-astra agent on the Franka FR3 rig with RealSense cameras.

The stock ``inspect-robots-franka`` adapter reads cameras over V4L2, which
is unreliable with several D435 units on one USB bus under real-time control.
A Python-driven run lets us inject the pyrealsense2 camera reader that the
openpi rig already runs stably.

Startup preserves the manually positioned arm and gripper; no home motion is
sent. Position the robot before launching, then confirm readiness in the console.

Usage (from the repo root, with the .venv active):

    python examples/run_franka.py examples/prompts/place_red_cube_in_green_cup.txt

The model endpoint and camera serials are read from
``~/.config/inspect-robots/config.ini`` and the constants in this file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from franka_rs_camera import make_rs_camera_reader

from inspect_robots import _dotenv, eval
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots_franka.embodiment import FrankaEmbodiment


class _ManualStartFrankaEmbodiment(FrankaEmbodiment):
    """Start from measured state without commanding the arm or gripper."""


    def _observe(self, instruction):
        from dataclasses import replace

        observation = super()._observe(instruction)
        depths = getattr(self._camera_reader, "depths", {})
        extra = dict(observation.extra)
        for name in observation.images:
            if name in depths:
                extra[f"{name}_depth"] = depths[name].copy()
        return replace(observation, extra=extra)

    def close(self):
        try:
            super().close()
        finally:
            close_reader = getattr(self._camera_reader, "close", None)
            if callable(close_reader):
                close_reader()

    def reset(self, scene: Scene, *, seed: int | None = None):
        import numpy as np

        if not callable(self._camera_reader):
            raise ValueError("manual-start mode requires an injected camera reader")
        if self._cfg.hostname is None:
            raise ValueError("Franka hostname is required")
        if not self._cfg.unattended:
            self._operator.wait_ready(
                "Keep the manually positioned arm and gripper as-is. "
                "Release manual control, enable FCI, then press Enter to start..."
            )
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)
        self._instruction = scene.instruction
        self.num_steps = 0
        observation = self._observe(scene.instruction)
        self._last_gripper_command = float(
            np.clip(
                self._driver.read_gripper_width() / self._cfg.gripper_max_width,
                0.0,
                1.0,
            )
        )
        self._t_last = self._clock()
        if not self._cfg.unattended:
            self._operator.output_fn(
                "Running from current pose: press Enter to end the episode, then y/N to score."
            )
        return observation


def _rtc_controller():
    """Build the RTC controller from ``INSPECT_ROBOTS_RTC*`` environment variables.

    Synchronous execution is the default: play the complete action chunk before
    requesting the next plan. Set ``INSPECT_ROBOTS_RTC=1`` to opt into RTC,
    which prefetches from a mid-chunk observation and may take over mid-chunk.
    Tuning knobs for that optional RTC mode (all optional):
    ``INSPECT_ROBOTS_RTC_PREFETCH`` (fraction consumed before prefetching),
    ``INSPECT_ROBOTS_RTC_HORIZON`` (seam-window index), ``INSPECT_ROBOTS_RTC_OVERLAP``
    (max blended fraction of the fresh tail), ``INSPECT_ROBOTS_RTC_SCHEDULE``
    (``zeros``/``ones``/``linear``/``exp``),
    ``INSPECT_ROBOTS_RTC_CONCURRENCY``
    (speculative requests per decision, default 1; each is a forked conversation
    and costs its own tokens/traffic), and ``INSPECT_ROBOTS_RTC_REANCHOR``
    (default on: rebuild the fresh absolute-target plan from the current pose
    instead of cross-fading, so a joint never reverses mid-motion and trips the
    arm's motion-generator discontinuity check).
    """
    enabled = os.environ.get("INSPECT_ROBOTS_RTC", "0").strip().lower()
    if enabled in {"0", "false", "off", "no"}:
        return None
    from inspect_robots_rtc import RTCConfig, RTCController, default_config

    base = default_config()
    reanchor = os.environ.get("INSPECT_ROBOTS_RTC_REANCHOR", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    concurrency = int(os.environ.get("INSPECT_ROBOTS_RTC_CONCURRENCY", "1"))
    return RTCController(
        RTCConfig(
            control_hz=15.0,
            execution_horizon=int(
                os.environ.get("INSPECT_ROBOTS_RTC_HORIZON", base.execution_horizon)
            ),
            schedule=os.environ.get("INSPECT_ROBOTS_RTC_SCHEDULE", base.schedule),  # type: ignore[arg-type]
            max_overlap_factor=float(
                os.environ.get("INSPECT_ROBOTS_RTC_OVERLAP", base.max_overlap_factor)
            ),
            prefetch_ratio=float(
                os.environ.get("INSPECT_ROBOTS_RTC_PREFETCH", base.prefetch_ratio)
            ),
        ),
        reanchor=reanchor,
        concurrency=concurrency,
    )


def _guardrails(embodiment, max_speed_frac: float, control_hz: float):
    """Clamp bounds and cap per-step motion to the agent's own speed budget.

    ``eval()`` defaults to a no-op approver, but a re-anchored RTC takeover can
    ask for a larger per-step change than the agent's toolset allowed. The cap
    mirrors the toolset's derived limit (``max_speed_frac / control_hz`` of each
    dimension's range, capped at its 5% backstop), so the arm never sees a swing
    large enough to trip libfranka's motion-generator discontinuity checks.
    """
    import numpy as np

    from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover

    space = embodiment.info.action_space
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    step_frac = min(max_speed_frac / control_hz, 0.05)
    max_delta = step_frac * (high - low)
    return ChainApprover(ClampApprover(space), DeltaLimitApprover(space, max_delta=max_delta))


def _read_policy_args() -> dict[str, str]:
    """Read the [policy.args] section for the agent into a flat dict."""
    import configparser

    cfg = configparser.ConfigParser()
    path = Path.home() / ".config" / "inspect-robots" / "config.ini"
    cfg.read(path)
    if not cfg.has_section("policy.args"):
        return {}
    return {k: v.strip() for k, v in cfg.items("policy.args")}


def _smooth_driver_factory(cfg):
    """Driver that moves joints with synchronous from-rest-to-rest motions.

    The stock adapter preempts a new JointMotion every control tick
    (`asynchronous=True`). With the agent emitting one absolute target per
    frame, that repeated preemption produces velocity/acceleration
    discontinuities that trip Franka's reflex protection. This driver instead
    issues a single blocking JointMotion per command, matching the openpi rig's
    ``set_joints`` semantics: each small move starts and ends at rest, so the
    command stream stays continuous and never triggers a reflex abort.
    """
    import contextlib
    import time

    import franky
    import numpy as np

    robot = franky.Robot(cfg.hostname)
    hand = franky.Gripper(cfg.hostname)
    robot.relative_dynamics_factor = cfg.relative_dynamics_factor

    _JOINT_TOL = 0.005  # radians; skip essentially-no-op moves
    # Below this target width, closing is a grasp attempt: use force-controlled
    # franky.grasp() (like openpi's rig, which grasps a 5 cm block at
    # width=0.05, force=15 N) so the fingers clamp on contact instead of just
    # stopping at a set width. Above it, treat as "open" and use move_async.
    _GRASP_WIDTH = cfg.gripper_max_width * 0.75
    _GRASP_FORCE = 15.0  # N, matches openpi utils.GRIPPER_FORCE
    _GRASP_EPSILON_OUTER = 0.1  # m, tolerates real block-size deviation
    # m; commands closer than this to the current width are treated as holds
    # and routed to move_async, never to grasp.
    _GRASP_OPEN_TOLLERANCE = 0.005
    # s; pause after aborting a pending gripper motion before the next command,
    # so libfranka accepts the new grasp instead of rejecting it mid-motion.
    _GRASP_SETTLE = 0.2
    # rad/s; joint speed below which the arm counts as stopped. franky's move()
    # returns when the motion generator reports finished, which can be before
    # the arm has actually decelerated. RTC can command a direction reversal at
    # a chunk seam; if the previous move still has residual velocity, libfranka
    # flags a velocity/acceleration discontinuity and aborts. Waiting for rest
    # before the next command keeps every rest-to-rest move continuous.
    _REST_VEL_EPS = 0.01
    # s; upper bound on that wait, so a stuck reading can never hang the loop.
    _REST_SETTLE_TIMEOUT = 1.0
    # Consecutive sub-eps velocity polls required before trusting "at rest", so
    # a single stale/settling sample cannot let the next move start early.
    _REST_STABLE_POLLS = 3

    class _SmoothDriver:
        def read_joints(self):
            return np.asarray(robot.current_joint_state.position)

        def read_gripper_width(self):
            return float(hand.width)


        def _wait_for_rest(self):
            deadline = time.monotonic() + _REST_SETTLE_TIMEOUT
            stable = 0
            while time.monotonic() < deadline:
                velocity = np.asarray(robot.current_joint_velocities)
                if float(np.max(np.abs(velocity))) < _REST_VEL_EPS:
                    stable += 1
                    if stable >= _REST_STABLE_POLLS:
                        return
                else:
                    stable = 0
                time.sleep(0.005)

        def _grasp(self, width):
            # Abort any pending, still-running open (or hold) command first:
            # closing while the hand is still moving aborts that motion and can
            # make libfranka reject the grasp with "Command failed!".
            with contextlib.suppress(franky.GripperException, franky.CommandException):
                hand.stop()
            time.sleep(_GRASP_SETTLE)
            hand.grasp(
                width,
                cfg.gripper_speed,
                _GRASP_FORCE,
                epsilon_outer=_GRASP_EPSILON_OUTER,
            )

        def move_joints(self, target):
            target = np.asarray(target)
            current = np.asarray(robot.current_joint_state.position)
            if np.max(np.abs(target - current)) < _JOINT_TOL:
                return
            try:
                robot.move(franky.JointMotion(target.tolist()))
            except (
                franky.CommandException,
                franky.ControlException,
                franky.NetworkException,
            ) as exc:
                # RTC runs inference traffic concurrently with control, so an
                # intermittent motion-generator/reflex/comm abort should not
                # kill the trial. Clear the fault, hold, and let the next
                # command resume from the measured pose.
                print(
                    f"[franka] recovered from {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                robot.recover_from_errors()
            self._wait_for_rest()

        def move_joints_sync(self, target):
            robot.move(franky.JointMotion(np.asarray(target).tolist()))
            self._wait_for_rest()

        def move_gripper(self, width):
            current = float(hand.width)
            closing = width <= current - _GRASP_OPEN_TOLLERANCE
            if closing and width <= _GRASP_WIDTH:
                # Force-controlled grasp with outer-contact detection, matching
                # the openpi grasp(width=0.05, force=15, epsilon_outer=0.1).
                # A plain move() to 5 cm just stops at the block with no grip
                # force, so the block can't be lifted.
                try:
                    self._grasp(width)
                except franky.GripperException:
                    # Recover the hand, then retry the grasp once.
                    with contextlib.suppress(franky.GripperException, franky.CommandException):
                        hand.stop()
                    time.sleep(_GRASP_SETTLE)
                    self._grasp(width)
            else:
                # Opening (or a same-width hold) must never re-enter grasp:
                # libfranka rejects grip commands while the hand is already
                # grasped or still moving, so widening always uses move_async.
                hand.move_async(width, cfg.gripper_speed)

        def disconnect(self):
            try:
                hand.stop()
                robot.stop()
            finally:
                pass

    return _SmoothDriver()


def _read_prompt(path: str | Path) -> str:
    """Read a non-empty UTF-8 task prompt from a text file."""
    prompt_path = Path(path).expanduser()
    try:
        instruction = prompt_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"cannot read UTF-8 prompt file {prompt_path}: {exc}") from exc
    if not instruction:
        raise SystemExit(f"prompt file is empty: {prompt_path}")
    return instruction


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit("usage: run_franka.py <prompt.txt>")
    if len(args) != 1:
        raise SystemExit("run_franka.py accepts one prompt text file path")
    prompt_path = Path(args[0]).expanduser()
    instruction = _read_prompt(prompt_path)

    _dotenv.init_dotenv(os.environ)  # load API keys from .env, like the CLI does

    import inspect_robots_franka.config as franka_config
    from inspect_robots_agent import LLMAgentPolicy
    from inspect_robots_franka.config import FrankaConfig

    # This rig has two exterior cameras and one wrist camera.
    franka_config.DEFAULT_CAMERAS = ("exterior_cam", "wrist_cam", "exterior_cam_2")

    policy_kvs = _read_policy_args()
    if "max_speed_frac" in policy_kvs:
        policy_kvs["max_speed_frac"] = float(policy_kvs["max_speed_frac"])
    # Send both exterior views and the wrist view captured by this rig.
    policy_kvs.setdefault("send_cameras", "exterior_cam,wrist_cam,exterior_cam_2")
    policy_kvs.setdefault("depth", "render")  # Attach aligned metric depth to model input.


    franka_cfg = FrankaConfig.from_kwargs(
        hostname="192.168.3.120",
        control_hz=15.0,
        relative_dynamics_factor=0.1,
        gripper_max_width=0.08,
        gripper_speed=0.03,
        cam_width=640,
        cam_height=480,
        rest_pose=None,  # No automatic parking motion after the task.
    )

    reader = make_rs_camera_reader(
        serials={
            "exterior_cam": "317622073233",
            "wrist_cam": "242322072531",
            "exterior_cam_2": "250122070653",
        },
        width=640,
        height=480,
        fps=15,
    )

    policy = LLMAgentPolicy(**policy_kvs)
    embodiment = _ManualStartFrankaEmbodiment(
        franka_cfg,
        camera_reader=reader,
        driver_factory=_smooth_driver_factory,
    )

    task = Task(
        name="franka-cube-grasp",
        scenes=[Scene(id="scene-0", instruction=instruction)],
        scorer=success_at_end(),
        max_steps=800,
    )

    controller = _rtc_controller()
    approver = _guardrails(embodiment, policy.config.max_speed_frac, franka_cfg.control_hz)
    print(f"prompt file: {prompt_path.resolve()}")
    print(f"running: {instruction}")
    print(f"policy args: {policy_kvs}")
    print(f"rtc: {'on' if controller is not None else 'off'}")
    try:
        (log,) = eval(task, policy, embodiment, controller=controller, approver=approver)
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            embodiment.close()
        if controller is not None:
            telemetry = controller.telemetry
            if telemetry is not None:
                print(
                    f"rtc telemetry: chunks={telemetry.chunks_fused} "
                    f"blends={telemetry.blends} delays={telemetry.delays}"
                )
    print(f"status: {log.status}")
    print(f"metrics: {log.results.metrics}")
    print(f"trial error: {log.error}")


if __name__ == "__main__":
    main()
