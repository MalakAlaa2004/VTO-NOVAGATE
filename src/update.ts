/**
 * ============================================================================
 *  Virtual Try-On (VTO) Tracking Engine — Three.js + MediaPipe FaceLandmarker
 * ============================================================================
 *
 *  Production Architecture (Instagram / Spark AR grade):
 *
 *  1. 3D ROTATION — MediaPipe PnP Face Transformation Matrix
 *     MediaPipe internally solves Perspective-n-Point (PnP) using its canonical
 *     3D face model, yielding a 4×4 rigid transformation matrix.
 *     This gives us the exact same rotation quality used by production AR SDKs
 *     (Instagram, Spark AR, Lens Studio). No manual Z-depth guessing needed.
 *
 *  2. POSITION — Multi-Landmark Camera-Ray Projection
 *     The 2D anchor point (blended inner + outer eye corners + nasion) is
 *     ray-cast through the Three.js PerspectiveCamera frustum. Depth is solved
 *     from the apparent inter-ocular distance vs known biometric width.
 *
 *  3. SCALE — Biometric Eye Distance Normalization
 *     The procedural model is built at 1:1 metric scale. Scale factor normalizes
 *     to the specific face's outer canthal width.
 *
 *  4. SMOOTHING — One-Euro Filter (Casiez et al. 2012)
 *     Separate filters for position, quaternion, and scale. Low minCutoff
 *     (0.08 Hz) pins the glasses rock-solid when stationary; high beta (2.5)
 *     opens the cutoff instantly during head motion for zero-latency tracking.
 *
 *  5. HEAD OCCLUSION PROXY — Depth-Only Cranial Ellipsoid
 *     Invisible geometry that writes only to the depth buffer, cleanly hiding
 *     temple arms behind the head in 3/4 and profile views.
 *
 *  6. ZERO HOT-PATH ALLOCATIONS
 *     All scratch vectors, matrices, and quaternions are module-scope singletons.
 */

import * as THREE from "three";
import type { Landmark, UpdateContext } from "./types";

// ============================================================================
// MediaPipe Canonical Landmark Indices
// ============================================================================
const LM_NASION        = 168; // Between the eyes (upper nose bridge)
const LM_GLABELLA      = 6;   // Between eyebrows
const LM_RIGHT_EYE_OUT = 33;  // Right eye outer corner
const LM_LEFT_EYE_OUT  = 263; // Left eye outer corner
const LM_RIGHT_EYE_IN  = 133; // Right eye inner corner (blink-immune)
const LM_LEFT_EYE_IN   = 362; // Left eye inner corner (blink-immune)

// Biometric reference (metric meters)
const REF_OUTER_CANTHAL_W = 0.092; // 92 mm average adult outer canthal width

// Placement fine-tuning (meters, in model-local coordinates)
const Y_OFFSET = -0.005; // Slight downward shift to sit on the nose bridge
const Z_OFFSET = 0.010;  // 10 mm forward clearance so lenses don't clip the nose

// ============================================================================
// One-Euro Filter (Casiez et al. 2012) — Production Jitter Elimination
// ============================================================================
class OneEuroFilter1D {
  private xPrev = 0;
  private dxPrev = 0;
  private initialized = false;

  constructor(
    private minCutoff: number = 0.08,
    private beta: number = 2.5,
    private dCutoff: number = 1.0
  ) {}

  public reset(): void { this.initialized = false; }

  public filter(x: number, dt: number): number {
    if (!this.initialized || dt <= 0 || dt > 0.5) {
      this.initialized = true;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxH = aD * dx + (1 - aD) * this.dxPrev;
    this.dxPrev = dxH;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxH);
    const a = this.alpha(cutoff, dt);
    const xH = a * x + (1 - a) * this.xPrev;
    this.xPrev = xH;
    return xH;
  }

  private alpha(fc: number, dt: number): number {
    const tau = 1.0 / (2.0 * Math.PI * fc);
    return 1.0 / (1.0 + tau / dt);
  }
}

class OneEuroFilterVec3 {
  private fx: OneEuroFilter1D;
  private fy: OneEuroFilter1D;
  private fz: OneEuroFilter1D;
  constructor(min = 0.08, beta = 2.5, dc = 1.0, minZ = 0.04, betaZ = 1.5) {
    this.fx = new OneEuroFilter1D(min, beta, dc);
    this.fy = new OneEuroFilter1D(min, beta, dc);
    this.fz = new OneEuroFilter1D(minZ, betaZ, dc); // Z (depth): extra stable
  }
  public reset(): void { this.fx.reset(); this.fy.reset(); this.fz.reset(); }
  public filter(v: THREE.Vector3, dt: number, out: THREE.Vector3): void {
    out.x = this.fx.filter(v.x, dt);
    out.y = this.fy.filter(v.y, dt);
    out.z = this.fz.filter(v.z, dt);
  }
}

class OneEuroFilterQuat {
  private qPrev = new THREE.Quaternion();
  private omegaPrev = 0;
  private initialized = false;
  constructor(
    private minCutoff: number = 0.08,
    private beta: number = 2.0,
    private dCutoff: number = 1.0
  ) {}
  public reset(): void { this.initialized = false; }

  public filter(q: THREE.Quaternion, dt: number, out: THREE.Quaternion): void {
    if (!this.initialized || dt <= 0 || dt > 0.5) {
      this.initialized = true;
      this.qPrev.copy(q);
      this.omegaPrev = 0;
      out.copy(q);
      return;
    }
    // Ensure shortest path
    if (this.qPrev.dot(q) < 0) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }

    const dot = Math.min(Math.max(this.qPrev.dot(q), -1), 1);
    const angle = 2 * Math.acos(dot);
    const omega = angle / dt;

    const tauD = 1 / (2 * Math.PI * this.dCutoff);
    const aD = 1 / (1 + tauD / dt);
    const omegaH = aD * omega + (1 - aD) * this.omegaPrev;
    this.omegaPrev = omegaH;

    const cutoff = this.minCutoff + this.beta * omegaH;
    const tau = 1 / (2 * Math.PI * cutoff);
    const alpha = 1 / (1 + tau / dt);

    this.qPrev.slerp(q, alpha);
    out.copy(this.qPrev);
  }
}

// Module-scope filter instances
const _posFilter   = new OneEuroFilterVec3(0.08, 2.5, 1.0, 0.04, 1.5);
const _quatFilter  = new OneEuroFilterQuat(0.08, 2.0, 1.0);
const _scaleFilter = new OneEuroFilter1D(0.04, 1.0, 0.8);

// ============================================================================
// Pre-allocated Scratch Objects (Zero Per-Frame Allocations)
// ============================================================================
const _ptAnchor    = new THREE.Vector3();
const _targetPos   = new THREE.Vector3();
const _targetQuat  = new THREE.Quaternion();
const _rotMatrix   = new THREE.Matrix4();
const _yAxis       = new THREE.Vector3();
const _zAxis       = new THREE.Vector3();

const _anchorLm: Landmark = { x: 0, y: 0, z: 0 };

// ============================================================================
// Head Occlusion Proxy (Stretch Goal #1)
// ============================================================================
let _occluderMesh: THREE.Mesh | null = null;

function ensureHeadOccluder(group: THREE.Group): void {
  if (_occluderMesh) return;
  const geo = new THREE.SphereGeometry(1, 24, 18);
  geo.scale(0.048, 0.075, 0.065);
  geo.translate(0, -0.012, -0.10);
  const mat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true });
  _occluderMesh = new THREE.Mesh(geo, mat);
  _occluderMesh.name = "head_occlusion_proxy";
  _occluderMesh.renderOrder = -1;
  group.add(_occluderMesh);
}

// ============================================================================
// Camera-Ray Projection (Landmark → World Position at Depth Z)
// ============================================================================
function landmarkToCameraRay(
  lm: Landmark,
  screenAspect: number,
  videoAspect: number,
  tanHalf: number,
  camAspect: number,
  out: THREE.Vector3,
): void {
  let sx = 1, sy = 1;
  if (screenAspect > videoAspect) sy = screenAspect / videoAspect;
  else sx = videoAspect / screenAspect;
  out.x = (lm.x - 0.5) * 2 * sx * tanHalf * camAspect;
  out.y = -((lm.y - 0.5) * 2 * sy) * tanHalf;
  out.z = -1;
}

// ============================================================================
// Rotation Extraction from MediaPipe's 4×4 Face Transformation Matrix
//
// The matrix transforms canonical face model → camera space.
// MediaPipe camera space: X-right, Y-down, Z-forward (into screen).
// Three.js world space:   X-right, Y-up,   Z-backward (out of screen).
//
// Coordinate conversion: R_three = M · R_mp · M⁻¹   where M = diag(1,−1,−1).
// Since M = M⁻¹ this simplifies to R_three = M · R_mp · M.
// ============================================================================
function extractRotation(data: number[], out: THREE.Quaternion): boolean {
  if (data.length < 16) return false;

  // MediaPipe data is row-major:
  //   Row 0: data[0..3],  Row 1: data[4..7],  Row 2: data[8..11]
  //
  // R_mp as 3×3:
  //   [data[0]  data[1]  data[2] ]
  //   [data[4]  data[5]  data[6] ]
  //   [data[8]  data[9]  data[10]]
  //
  // R_three = M · R_mp · M   (M = diag(1, -1, -1)):
  //   [ data[0]  -data[1]  -data[2] ]
  //   [-data[4]   data[5]   data[6] ]
  //   [-data[8]   data[9]   data[10]]

  // Three.js Matrix4.set() takes ROW-MAJOR arguments: (n11,n12,n13,n14, ...)
  _rotMatrix.set(
     data[0], -data[1], -data[2],  0,
    -data[4],  data[5],  data[6],  0,
    -data[8],  data[9],  data[10], 0,
     0,        0,        0,        1,
  );
  out.setFromRotationMatrix(_rotMatrix);
  return true;
}

// ============================================================================
// Main Update Routine
// ============================================================================
export function update(landmarks: Landmark[], ctx: UpdateContext): void {
  const { sunglasses, camera, videoWidth, videoHeight, dt, faceMatrix } = ctx;

  ensureHeadOccluder(sunglasses);

  if (!landmarks || landmarks.length < 468) return;

  const lmNasion   = landmarks[LM_NASION];
  const lmGlabella = landmarks[LM_GLABELLA];
  const lmROut     = landmarks[LM_RIGHT_EYE_OUT];
  const lmLOut     = landmarks[LM_LEFT_EYE_OUT];
  const lmRIn      = landmarks[LM_RIGHT_EYE_IN];
  const lmLIn      = landmarks[LM_LEFT_EYE_IN];

  if (!lmNasion || !lmGlabella || !lmROut || !lmLOut || !lmRIn || !lmLIn) return;

  // Camera / video geometry
  const tanHalf    = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const camAspect  = camera.aspect;
  const screenAR   = window.innerWidth / Math.max(window.innerHeight, 1);
  const videoAR    = videoWidth > 0 && videoHeight > 0 ? videoWidth / videoHeight : screenAR;

  // --------------------------------------------------------------------------
  // 1. ROTATION — from MediaPipe PnP matrix
  // --------------------------------------------------------------------------
  let hasRotation = false;
  if (faceMatrix) {
    hasRotation = extractRotation(faceMatrix, _targetQuat);
  }

  if (!hasRotation) {
    // Fallback: simple landmark-based rotation (less accurate but functional)
    const dx = lmLOut.x - lmROut.x;
    const dy = lmLOut.y - lmROut.y;
    const roll = -Math.atan2(dy, dx);
    _targetQuat.setFromAxisAngle(_yAxis.set(0, 1, 0), 0);
    _targetQuat.multiply(_targetQuat.clone().setFromAxisAngle(_zAxis.set(0, 0, 1), roll));
  }

  // --------------------------------------------------------------------------
  // 2. DEPTH — from apparent inter-ocular distance
  // --------------------------------------------------------------------------
  // Project eye corners to camera rays at unit depth
  landmarkToCameraRay(lmROut, screenAR, videoAR, tanHalf, camAspect, _ptAnchor);
  const rxOut = _ptAnchor.x;
  landmarkToCameraRay(lmLOut, screenAR, videoAR, tanHalf, camAspect, _ptAnchor);
  const lxOut = _ptAnchor.x;
  const ryOut = _ptAnchor.y; // reuse last y

  landmarkToCameraRay(lmROut, screenAR, videoAR, tanHalf, camAspect, _ptAnchor);
  const ryOut2 = _ptAnchor.y;

  const deltaRayOuter = Math.abs(lxOut - rxOut);
  const targetDepth = THREE.MathUtils.clamp(
    REF_OUTER_CANTHAL_W / Math.max(deltaRayOuter, 0.001),
    0.20, 2.50,
  );

  // --------------------------------------------------------------------------
  // 3. POSITION — anchor at blended eye-center / nasion
  // --------------------------------------------------------------------------
  const outerMidX = (lmROut.x + lmLOut.x) * 0.5;
  const innerMidX = (lmRIn.x + lmLIn.x) * 0.5;
  _anchorLm.x = outerMidX * 0.5 + innerMidX * 0.5;

  const outerMidY = (lmROut.y + lmLOut.y) * 0.5;
  const innerMidY = (lmRIn.y + lmLIn.y) * 0.5;
  const eyeY = outerMidY * 0.5 + innerMidY * 0.5;
  const bridgeY = lmNasion.y * 0.7 + lmGlabella.y * 0.3;
  _anchorLm.y = bridgeY * 0.55 + eyeY * 0.45;

  _anchorLm.z = 0;

  landmarkToCameraRay(_anchorLm, screenAR, videoAR, tanHalf, camAspect, _ptAnchor);
  _targetPos.copy(_ptAnchor).multiplyScalar(targetDepth);

  // Apply offsets in model-local coordinate frame (using rotation axes)
  _yAxis.set(0, 1, 0).applyQuaternion(_targetQuat);
  _zAxis.set(0, 0, 1).applyQuaternion(_targetQuat);
  _targetPos.addScaledVector(_yAxis, Y_OFFSET);
  _targetPos.addScaledVector(_zAxis, Z_OFFSET);

  // --------------------------------------------------------------------------
  // 4. SCALE — proportional to face width
  // --------------------------------------------------------------------------
  const dxEye = lmLOut.x - lmROut.x;
  const dyEye = lmLOut.y - lmROut.y;
  const eyeSpanNorm = Math.sqrt(dxEye * dxEye + dyEye * dyEye);
  // At reference distance (~0.6m), eye span in normalized coords ≈ 0.15
  const rawScale = (eyeSpanNorm / 0.15) * 1.0;
  const targetScale = THREE.MathUtils.clamp(rawScale, 0.70, 1.40);

  // --------------------------------------------------------------------------
  // 5. SMOOTHING — One-Euro Filter
  // --------------------------------------------------------------------------
  _posFilter.filter(_targetPos, dt, sunglasses.position);
  _quatFilter.filter(_targetQuat, dt, sunglasses.quaternion);
  sunglasses.scale.setScalar(_scaleFilter.filter(targetScale, dt));

  sunglasses.visible = true;
}
