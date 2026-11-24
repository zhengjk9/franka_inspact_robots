# Franka FR3 rig — current setup

- One Franka FR3 arm with a parallel-jaw gripper.
- One wrist camera on the hand, slightly tilted so the gripper and the space
  between the fingers are clearly visible. Use it to confirm the block is
  below and between the fingers before closing.
- Two exterior cameras: one at the left-rear and one at the right-rear of the
  workspace, giving two different views of the arm over the table.
- Camera streams in each observation:
  - `wrist_cam`: the wrist view described above.
  - `exterior_cam`: the exterior camera on one side.
  - `exterior_cam_2`: the exterior camera on the opposite side.
- Judge alignment with the two exterior views together: the wrist view can
  look "large and centered" while the gripper is actually beside or above the
  block, so prefer the exterior views for reach and left/right alignment.

;; ## Current task: open the drawer

;; - Goal: pull open the drawer directly in front of the robot.
;; - The start pose already faces the drawer, so the handle is straight ahead at
;;   roughly the tool's height; no base reorientation should be needed.
;; - Sequence: extend the open jaws forward onto the handle, close to grip it,
;;   then pull straight back to slide the drawer open.
;; - Reach forward with joint4 (less negative extends forward, as above) and
;;   adjust height with joint2; keep the wrist view on the handle while reaching
;;   so the jaws stay centered.
;; - Once the jaws straddle the handle, close the gripper and confirm it stalls on
;;   the handle rather than closing fully. Then pull back with joint4 (more
;;   negative) until the drawer is open.

## Joint movement directions (measured by forward kinematics at the earlier start posture joint2 +0.18 / joint4 -2.21; the current home is joint2 +0.269 / joint4 -2.321, so re-derive if a direction looks off)

- joint1 (base yaw): swings the whole arm horizontally, left/right in the
  exterior views. The tool height does not change. Use it to center the tool
  over the block.
- joint2 (shoulder pitch): moving joint2 negative raises the tool; moving it
  positive lowers it. From the start posture, joint2 -0.5 raises the tool
  ~0.42 m while the reach stays close, and moving it to -1.0 raises it ~0.66 m
  and swings it well back toward the base. Either way it tilts the gripper
  heavily, so re-correct tilt with joint6.
- joint3 (upper-arm roll): swings the tool horizontally, left/right, with
  essentially no height change at this posture. joint1 and joint3 together aim
  the tool across the table.
- joint4 (elbow pitch): with the elbow raised, making joint4 less negative
  both raises the tool and extends it forward (joint4 -1.6 reaches ~0.15 m
  further forward and ~0.35 m higher than -2.4). Making it more negative
  lowers the tool and pulls it back (~0.08 m per -0.2 step). To land on a
  block, lower with joint4 and compensate the backward drift with joint3 or
  joint2.
- joint5 (forearm roll): rotates the forearm around its own axis; the tool
  position barely changes. With the wrist bent, rotating it also tilts the
  gripper off vertical.
- joint6 (wrist pitch): tilts the gripper forward/back. Use it to reduce a
  gripper tilt visible in the wrist view, in whichever direction makes the
  fingers point more vertically; the required direction changes as the arm
  postures change.
- joint7 (hand roll): rotates the hand around the tool axis; the tool
  position does not change. Move it only if the wrist view shows the fingers
  twisted.
- Caution: lowering with joint4 alone both tilts the gripper off vertical and
  drifts the tool backward. Moving joint2 strongly negative raises the tool
  but tilts it heavily. Correct the tilt with joint6 after, or during, the
  descent by watching the wrist view.
- These directions are pinned to the start posture above. At a deeply lowered
  posture (joint2 ≈ -0.8) the reach behavior of joint2/joint4 inverts: there,
  joint4 less negative raises with reach nearly unchanged, and joint2 negative
  also swings the tool backward. Re-derive from the wrist and exterior views
  rather than trusting the sign of a single joint.

## Gripper behaviour

- The gripper slot is normalized: 0 = fully closed, 1 = open. The observed
  value is the physical finger width divided by 0.08 m, clipped to [0, 1].
- The hand's maximum opening is whatever it physically reaches on a given
  boot. A fully-open command can read back well below 1.0 (for example ~0.28)
  when the hand is homed to a smaller range. Treat the observed value as the
  truth; the slot does not need to read 1.0 for the hand to be open.
- After you command the gripper, wait one observation for it to settle. If a
  re-issued command changes nothing, the gripper is settled: do not keep
  re-opening it or theorize about lag or stuck state. Read the wrist camera to
  confirm the fingers are clear of the block, then move on.
- Issue at most one gripper command per observation, and never close while the
  observed width is still changing (for example right after an open that read
  back only 0.9). Grasping a hand that is still opening aborts that motion and
  makes the grasp command fail mid-episode.