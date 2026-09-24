# AR Virtual Try-On — Interview Sandbox

A small browser app that streams your webcam, runs MediaPipe FaceLandmarker on
each frame, and renders a three.js scene on top. Your job is to make a pair of
procedural sunglasses track the wearer's face.

---

## 1. Setup

**Requirements**

- Node.js ≥ 18
- A webcam
- A modern Chromium- or Firefox-based browser. Safari works but FaceLandmarker
  is noticeably slower.

**Install and run**

```bash
npm install
npm run dev
```

Open the URL Vite prints (defaults to `http://localhost:5173`).
`getUserMedia` requires HTTPS or localhost — both `localhost` and `127.0.0.1`
are fine, anything else needs HTTPS.

**Typecheck**

```bash
npm run typecheck
```

`strict` mode is on. Please keep it green.

---

## 2. What's provided

```
ar-vto-sandbox/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── src/
    ├── main.ts          # scene, camera, mediapipe, render loop  ← rarely needs edits
    ├── sunglasses.ts    # procedural sunglasses model
    ├── update.ts        # ←★ YOU EDIT THIS ★
    ├── types.ts         # Landmark, UpdateContext
    └── style.css
```

`main.ts` already does:

- Webcam capture rendered behind the canvas as a mirrored `<video>` element.
- A `THREE.PerspectiveCamera` (45° vertical FOV, you may tune) at the origin
  looking down `-Z`.
- A `THREE.Scene` with ambient + key + rim lighting and ACES tone mapping.
- A `sunglasses` `THREE.Group` already added to the scene. The model is
  procedural and to real-world scale (1 unit ≈ 1 meter). Its pivot is at the
  **center of the bridge**, with:
  - `+X` → wearer's right
  - `+Y` → up
  - `+Z` → out of the face toward the camera
- MediaPipe FaceLandmarker running in `VIDEO` mode on the GPU delegate,
  yielding 478 landmarks per detected frame.
- An HUD showing render fps and detection state.
- A landmark debug overlay toggled with **`D`** that renders all 478 points
  as small green dots — useful while debugging.

Each detected frame, `main.ts` calls:

```ts
update(landmarks, ctx);
```

with:

```ts
type Landmark = { x: number; y: number; z: number };

interface UpdateContext {
  sunglasses: THREE.Group;
  camera: THREE.PerspectiveCamera;
  scene: THREE.Scene;
  videoWidth: number;
  videoHeight: number;
  dt: number; // seconds since the previous detected frame
}
```

---

## 3. The task

In `src/update.ts`, position, scale, and orient `ctx.sunglasses` so it sits
naturally on the wearer's face. Specifically:

1. **Position** — the bridge of the glasses on the bridge of the nose,
   between the eyes.
2. **Scale** — proportional to face width (pupillary distance / temple-to-
   temple), so the glasses look right whether the user is close to or far
   from the camera.
3. **Orientation** — track yaw (look left/right), pitch (nod), and roll
   (head tilt). Profile view should also look right.
4. **Smoothing** — stable across frames without visible lag. Raw landmarks
   jitter; naïvely copying them every frame produces a "swimmy" look.

You should not need to touch `main.ts` or `sunglasses.ts` to complete the
core task, but you may if you have a reason. Explain it if you do.

### Coordinate-system gotcha

Landmark `x` and `y` are normalized **image-space** coordinates (top-left
origin, y increases downward). The three.js camera renders into world space
with y-up. You cannot just write `sunglasses.position.set(lm.x, lm.y, lm.z)`
and expect it to work — those numbers live in different spaces, and `z` is a
relative depth in roughly image-space units, not meters.

You will need to think about how to project landmarks into the world such
that the glasses, when rendered through the camera, line up with the face
visible behind them in the video.

### Useful landmark indices

The FaceLandmarker uses the [canonical FaceMesh topology](https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png).
Some indices you'll likely reach for:

| Index | Location                       |
| ----- | ------------------------------ |
| 1     | Nose tip                       |
| 4     | Mid nose bridge                |
| 6     | Glabella (between eyebrows)    |
| 168   | Between the eyes               |
| 33    | Right eye, outer corner        |
| 133   | Right eye, inner corner        |
| 362   | Left eye, inner corner         |
| 263   | Left eye, outer corner         |
| 10    | Forehead midpoint              |
| 152   | Chin                           |
| 234   | Right cheekbone (face width)   |
| 454   | Left cheekbone (face width)    |

A few of these are common starting points for an eyewear anchor. You don't
have to use any of them — pick what works.

---

## 4. Hints (read if stuck)

You don't need any of these to start. They are here so you don't get blocked
on a single corner.

- **Anchor.** The bridge of the nose between the eyes (~168, or the midpoint
  of 33/263) is a stable anchor.
- **Scale.** A robust scale signal is interocular or temple-to-temple
  distance in image space. Convert that to a world-space scale that's
  consistent with however you project the anchor into the world.
- **Orientation.** Three landmarks define a rigid basis:
  - take the vector from right-eye to left-eye as the X axis,
  - cross it with the vector from the bridge to a forehead/chin landmark
    to get Z,
  - cross those back for a clean orthonormal basis (Y),
  - then `Matrix4.makeBasis(x, y, z)` or convert to a quaternion.
  Build rotation this way; do **not** derive Euler angles from chained
  `atan2` calls.
- **Projection.** One approach: unproject the 2D anchor through the camera
  to get a ray, then place the glasses along that ray at a depth chosen so
  the rendered face width matches the measured face width in pixels. There
  are other valid approaches. State your assumptions either way.
- **Smoothing.** A One-Euro filter (Casiez et al.) gives a great
  jitter/latency trade-off and is ~30 lines. An EMA / lerp toward target
  with α ≈ 0.3–0.5 is a fine baseline. Smooth position, scale, and a
  quaternion (slerp) separately. Naming what you used and why is more
  important than choosing the "best" one.
- **No allocations in the hot path.** Hoist `Vector3`/`Quaternion`/`Matrix4`
  scratch objects to module scope and reuse them. The stub in `update.ts`
  shows where.

---

## 5. Stretch goals

In order of difficulty. Pick one or two if you finish the core task with
time to spare; we'd rather see one done well than three half-finished.

1. **Occlusion** — render an invisible head proxy that writes to the depth
   buffer, so the temple arms get correctly hidden behind the head when the
   wearer turns to profile.
2. **Inter-frame interpolation** — keep the render loop at 60 fps even if
   detection only fires at 15–20 fps, by interpolating pose between
   detections.
3. **IPD calibration** — handle users with different pupillary distances
   without the glasses looking comically wide or narrow.
4. **Recovery** — graceful behaviour when the face leaves frame and
   returns (no snap, no NaN, no rubber-band).
5. **Lens reflection** — a simple environment map on the lenses so they
   read as glass under motion.

---

## 6. What we're evaluating

- **Correctness.** Does it look right at neutral, profile, tilt, close,
  and far?
- **Coordinate-system reasoning.** Can you explain how landmark space maps
  to world space, and why?
- **3D math.** Clean construction of a rotation basis. No gimbal traps.
- **three.js fluency.** Idiomatic use of `Quaternion`, `Matrix4`,
  `Object3D` transforms, projection. Awareness of color space and depth.
- **Performance hygiene.** No per-frame allocations. Smoothing chosen
  deliberately, not by feel.
- **TypeScript hygiene.** No `any`. Strict mode stays green.
- **Communication.** During the session: think out loud. After: be ready
  to walk through your trade-offs.

---

## 7. Time budget

- Live pairing: **75 minutes** for the core task, followed by ~15 minutes
  of discussion.
- Take-home: **3–4 hours**, including any stretch goals you choose.

If you spend more than ~20 minutes blocked on a single thing, ask (live)
or write it down as a note and move on (take-home).

---

## 8. Troubleshooting

- **Black screen, no webcam.** Check the page is on `localhost` or HTTPS.
  Confirm the camera permission prompt was granted. Some OSes also need
  app-level camera permission for the browser.
- **`face: searching…` never turns green.** Make sure your face is well-
  lit and roughly centred. The first detection can take a second or two
  while the wasm and model download.
- **Glasses are upside-down or mirrored.** Remember the canvas is mirrored
  in CSS (`scaleX(-1)`) to match the video. You may need to account for
  the handedness flip in your basis construction.
- **fps tanks when you move.** Probably allocating in the hot path or
  re-uploading geometry. Check `update.ts` for `new` inside the function.
- **`detectForVideo` throws.** The MediaPipe wasm or `.task` model fetch
  failed — check the network tab. Both come from public CDNs.

Good luck. Have fun with it.
