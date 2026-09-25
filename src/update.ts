/**
 * ============================================================================
 *  Virtual Try-On (VTO) Tracking Engine — Three.js + MediaPipe FaceLandmarker
 * ============================================================================
 *
 *  Senior Graphics / AR Architecture:
 *
 *  1. RIGID 3D ORTHONORMAL HEAD POSE (Direct Landmark Solution)
 *     Instead of relying on the buggy MediaPipe facialTransformationMatrix
 *     (which has a known upstream bug causing sideways skew/rotations),
 *     we construct a pure, mathematically rigid orthonormal basis [X, Y, Z]
 *     directly from anatomically stable cephalometric landmarks:
 *       - X (Lateral):    Averaged outer + inner canthi + temples (Right → Left)
 *       - Z (Sagittal):   Frankfurt plane vector from mid-temples to nasal bridge
 *                         Gram-Schmidt orthogonalized against X (Out of face)
 *       - Y (Vertical):   Strict right-handed cross product Z × X (Up along skull)
 *     This guarantees det(R) = +1, zero shear, zero distortion, and perfect
 *     alignment of temple arms back towards the ears without lateral skew.
 *
 *  2. CAMERA-RAY ANCHORING
 *     The 2D nasal bridge anchor is projected through the Three.js PerspectiveCamera
 *     frustum, accounting for CSS object-fit: cover aspect scaling. The anchor
 *     is pinned rock-solid between the inner eye corners and nasion.
 *
 *  3. METRIC DEPTH WITH FORESHORTENING COMPENSATION
 *     Depth is computed from the apparent outer canthal distance (92 mm standard
 *     adult baseline) compensated by cos(yaw) foreshortening:
 *       depth = (0.092 * cos(yaw)) / deltaRayOuter
 *     This prevents the glasses from flying backward / "breathing" during head turns.
 *
 *  4. TEMPORAL STABILITY — One-Euro Filter (Casiez et al. 2012)
 *     Separate adaptive filters for position, quaternion, and scale.
 *     Zero jitter when stationary, zero lag during head motion.
 *
 *  5. HEAD OCCLUSION PROXY
 *     Invisible cranial geometry writes only to the depth buffer, cleanly hiding
 *     the temple arms behind the head in 3/4 and profile views.
 *
 *  6. ZERO HOT-PATH ALLOCATIONS
 *     All scratch vectors, matrices, and quaternions are pre-allocated module singletons.
 */

import * as THREE from "three";
import type { Landmark, UpdateContext } from "./types";

// ============================================================================
// MediaPipe Canonical Landmark Indices
// ============================================================================
const LM_NASION        = 168; // Nasal root between eyes
const LM_GLABELLA      = 6;   // Between eyebrows
const LM_RIGHT_EYE_OUT = 33;  // Right eye outer corner (canthus)
const LM_LEFT_EYE_OUT  = 263; // Left eye outer corner (canthus)
const LM_RIGHT_EYE_IN  = 133; // Right eye inner corner (blink-immune)
const LM_LEFT_EYE_IN   = 362; // Left eye inner corner (blink-immune)
const LM_RIGHT_TEMPLE  = 234; // Right side temple / tragus level
const LM_LEFT_TEMPLE   = 454; // Left side temple / tragus level

// Biometric reference dimensions (metric meters)
const REF_OUTER_CANTHAL_W = 0.092; // 92 mm average adult outer canthal distance

// Fine-tuning placement (meters, in head-local coordinate frame)
const Y_OFFSET = -0.005; // Placed on upper nasal bridge so lenses align with pupils
const Z_OFFSET =  0.012; // 12 mm forward clearance to rest gently without clipping nose

// ============================================================================
// One-Euro Filter (Casiez et al. 2012) — Production Jitter Elimination
// ============================================================================
class OneEuroFilter1D {
  private xPrev = 0;
  private dxPrev = 0;
  private initialized = false;

  constructor(
    private minCutoff: number = 0.10,
    private beta: number = 3.0,
    private dCutoff: number = 1.0,
  ) {}

  public reset(): void {
    this.initialized = false;
  }

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

  constructor(min = 0.10, beta = 3.0, dc = 1.0) {
    this.fx = new OneEuroFilter1D(min, beta, dc);
    this.fy = new OneEuroFilter1D(min, beta, dc);
    this.fz = new OneEuroFilter1D(min * 0.5, beta * 0.75, dc); // Z extra damped against webcam noise
  }

  public reset(): void {
    this.fx.reset();
    this.fy.reset();
    this.fz.reset();
  }

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
    private minCutoff: number = 0.10,
    private beta: number = 2.5,
    private dCutoff: number = 1.0,
  ) {}

  public reset(): void {
    this.initialized = false;
  }

  public filter(q: THREE.Quaternion, dt: number, out: THREE.Quaternion): void {
    if (!this.initialized || dt <= 0 || dt > 0.5) {
      this.initialized = true;
      this.qPrev.copy(q);
      this.omegaPrev = 0;
      out.copy(q);
      return;
    }

    // Shortest path interpolation
    if (this.qPrev.dot(q) < 0) {
      q.x = -q.x;
      q.y = -q.y;
      q.z = -q.z;
      q.w = -q.w;
    }

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
const _posFilter   = new OneEuroFilterVec3(0.10, 3.0, 1.0);
const _quatFilter  = new OneEuroFilterQuat(0.10, 2.5, 1.0);
const _scaleFilter = new OneEuroFilter1D(0.04, 1.0, 0.8);

// ============================================================================
// Pre-allocated Scratch Objects (Zero Garbage Collection)
// ============================================================================
const _ptRightEyeOut = new THREE.Vector3();
const _ptLeftEyeOut  = new THREE.Vector3();
const _ptRightEyeIn  = new THREE.Vector3();
const _ptLeftEyeIn   = new THREE.Vector3();
const _ptRightTemple = new THREE.Vector3();
const _ptLeftTemple  = new THREE.Vector3();
const _ptNasion      = new THREE.Vector3();
const _ptGlabella    = new THREE.Vector3();

const _vEyes         = new THREE.Vector3();
const _vTemples      = new THREE.Vector3();
const _vLat          = new THREE.Vector3();
const _ptMidTemples  = new THREE.Vector3();
const _ptMidNose     = new THREE.Vector3();
const _vFwd          = new THREE.Vector3();

const _xAxis         = new THREE.Vector3();
const _yAxis         = new THREE.Vector3();
const _zAxis         = new THREE.Vector3();
const _basisMatrix   = new THREE.Matrix4();

const _rayRightOut   = new THREE.Vector3();
const _rayLeftOut    = new THREE.Vector3();
const _rayAnchor     = new THREE.Vector3();
const _targetPos     = new THREE.Vector3();
const _targetQuat    = new THREE.Quaternion();

// ============================================================================
// Head Occlusion Proxy (Hides temple arms behind head in 3D profile views)
// ============================================================================
let _occluderMesh: THREE.Mesh | null = null;

function ensureHeadOccluder(group: THREE.Group): void {
  if (_occluderMesh) return;
  const geo = new THREE.SphereGeometry(1, 24, 18);
  geo.scale(0.065, 0.090, 0.080);
  geo.translate(0, -0.015, -0.10);
  const mat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true });
  _occluderMesh = new THREE.Mesh(geo, mat);
  _occluderMesh.name = "head_occlusion_proxy";
  _occluderMesh.renderOrder = -1;
  group.add(_occluderMesh);
}

// ============================================================================
// Coordinate Helpers
// ============================================================================
/**
 * Convert MediaPipe landmark to isotropic 3D point (metric proportioned).
 * MediaPipe coordinate rules:
 *   - X: 0 (left) to 1 (right)
 *   - Y: 0 (top) to 1 (bottom)  → In Three.js, Y is UP, so we negate
 *   - Z: roughly on same metric scale as X, negative closer to camera
 *        In Three.js (looking down -Z), closer to camera means LARGER Z, so negate
 */
function lmToIsotropic3D(lm: Landmark, videoAspect: number, out: THREE.Vector3): void {
  out.x = lm.x;
  out.y = -lm.y / videoAspect;
  out.z = -lm.z;
}

/**
 * Shoot camera ray for landmark (accounting for CSS object-fit: cover scaling).
 */
function landmarkToCameraRay(
  x: number,
  y: number,
  screenAspect: number,
  videoAspect: number,
  tanHalfFov: number,
  camAspect: number,
  out: THREE.Vector3,
): void {
  let sx = 1;
  let sy = 1;
  if (screenAspect > videoAspect) {
    sy = screenAspect / videoAspect;
  } else {
    sx = videoAspect / screenAspect;
  }
  out.x = (x - 0.5) * 2 * sx * tanHalfFov * camAspect;
  out.y = -((y - 0.5) * 2 * sy) * tanHalfFov;
  out.z = -1;
}

// ============================================================================
// Main Update Routine
// ============================================================================
export function update(landmarks: Landmark[], ctx: UpdateContext): void {
  const { sunglasses, camera, videoWidth, videoHeight, dt } = ctx;

  ensureHeadOccluder(sunglasses);

  if (!landmarks || landmarks.length < 468) {
    return;
  }

  const lmNasion      = landmarks[LM_NASION];
  const lmGlabella    = landmarks[LM_GLABELLA];
  const lmRightEyeOut = landmarks[LM_RIGHT_EYE_OUT];
  const lmLeftEyeOut  = landmarks[LM_LEFT_EYE_OUT];
  const lmRightEyeIn  = landmarks[LM_RIGHT_EYE_IN];
  const lmLeftEyeIn   = landmarks[LM_LEFT_EYE_IN];
  const lmRightTemple = landmarks[LM_RIGHT_TEMPLE];
  const lmLeftTemple  = landmarks[LM_LEFT_TEMPLE];

  if (
    !lmNasion || !lmGlabella ||
    !lmRightEyeOut || !lmLeftEyeOut ||
    !lmRightEyeIn || !lmLeftEyeIn ||
    !lmRightTemple || !lmLeftTemple
  ) {
    return;
  }

  // Camera & viewport geometry
  const tanHalfFov   = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const camAspect    = camera.aspect;
  const screenAspect = window.innerWidth / Math.max(window.innerHeight, 1);
  const videoAspect  = videoWidth > 0 && videoHeight > 0 ? videoWidth / videoHeight : screenAspect;

  // --------------------------------------------------------------------------
  // 1. 3D ROTATION — Pure Cephalometric Orthonormal Basis
  // --------------------------------------------------------------------------
  lmToIsotropic3D(lmRightEyeOut, videoAspect, _ptRightEyeOut);
  lmToIsotropic3D(lmLeftEyeOut,  videoAspect, _ptLeftEyeOut);
  lmToIsotropic3D(lmRightEyeIn,  videoAspect, _ptRightEyeIn);
  lmToIsotropic3D(lmLeftEyeIn,   videoAspect, _ptLeftEyeIn);
  lmToIsotropic3D(lmRightTemple, videoAspect, _ptRightTemple);
  lmToIsotropic3D(lmLeftTemple,  videoAspect, _ptLeftTemple);
  lmToIsotropic3D(lmNasion,      videoAspect, _ptNasion);
  lmToIsotropic3D(lmGlabella,    videoAspect, _ptGlabella);

  // X-Axis (Lateral / Right-to-Left):
  // Weighted blend of inner/outer eye canthi + temples gives max stability
  _vEyes.subVectors(_ptLeftEyeOut, _ptRightEyeOut).addScaledVector(
    _vLat.subVectors(_ptLeftEyeIn, _ptRightEyeIn),
    1.0,
  ).multiplyScalar(0.5);

  _vTemples.subVectors(_ptLeftTemple, _ptRightTemple);

  _xAxis.copy(_vEyes).multiplyScalar(0.65).addScaledVector(_vTemples, 0.35);
  if (_xAxis.lengthSq() < 1e-6) return;
  _xAxis.normalize();

  // Z-Axis (Forward / Sagittal — Out of the face):
  // Vector from midpoint of temples to upper nasal bridge (Frankfurt plane)
  _ptMidTemples.addVectors(_ptRightTemple, _ptLeftTemple).multiplyScalar(0.5);
  _ptMidNose.copy(_ptNasion).multiplyScalar(0.7).addScaledVector(_ptGlabella, 0.3);
  _vFwd.subVectors(_ptMidNose, _ptMidTemples);

  // Gram-Schmidt orthogonalization: ensure Z is strictly perpendicular to X
  _zAxis.copy(_vFwd).addScaledVector(_xAxis, -_vFwd.dot(_xAxis));
  if (_zAxis.lengthSq() < 1e-6) return;
  _zAxis.normalize();

  // Y-Axis (Vertical / Superior):
  // Right-handed orthonormal cross product: Y = Z × X
  _yAxis.crossVectors(_zAxis, _xAxis).normalize();

  // Construct pure rotation matrix with det = +1
  _basisMatrix.makeBasis(_xAxis, _yAxis, _zAxis);
  _targetQuat.setFromRotationMatrix(_basisMatrix);

  // --------------------------------------------------------------------------
  // 2. DEPTH — Metric Outer Canthal Inter-Ocular Projection
  // --------------------------------------------------------------------------
  landmarkToCameraRay(lmRightEyeOut.x, lmRightEyeOut.y, screenAspect, videoAspect, tanHalfFov, camAspect, _rayRightOut);
  landmarkToCameraRay(lmLeftEyeOut.x,  lmLeftEyeOut.y,  screenAspect, videoAspect, tanHalfFov, camAspect, _rayLeftOut);

  const dxRay = _rayLeftOut.x - _rayRightOut.x;
  const dyRay = _rayLeftOut.y - _rayRightOut.y;
  const deltaRayOuter = Math.sqrt(dxRay * dxRay + dyRay * dyRay);

  // Yaw foreshortening factor: cos(yaw) = sqrt(1 - xAxis.z^2)
  // Prevents the glasses from flying backward when turning head sideways
  const cosYaw = Math.max(Math.sqrt(Math.max(0.01, 1.0 - _xAxis.z * _xAxis.z)), 0.35);

  const rawDepth = (REF_OUTER_CANTHAL_W * cosYaw) / Math.max(deltaRayOuter, 0.001);
  const targetDepth = THREE.MathUtils.clamp(rawDepth, 0.25, 2.50);

  // --------------------------------------------------------------------------
  // 3. POSITION — Multi-Landmark Nasal Bridge Anchor
  // --------------------------------------------------------------------------
  // Anchor horizontally at mid-eye / nasion line
  const innerMidX = (lmRightEyeIn.x + lmLeftEyeIn.x) * 0.5;
  const outerMidX = (lmRightEyeOut.x + lmLeftEyeOut.x) * 0.5;
  const anchorX   = innerMidX * 0.45 + outerMidX * 0.25 + lmNasion.x * 0.30;

  // Anchor vertically at nasal bridge (pupil / eye level)
  const innerMidY = (lmRightEyeIn.y + lmLeftEyeIn.y) * 0.5;
  const outerMidY = (lmRightEyeOut.y + lmLeftEyeOut.y) * 0.5;
  const eyeLevelY = innerMidY * 0.5 + outerMidY * 0.5;
  const anchorY   = eyeLevelY * 0.55 + lmNasion.y * 0.45;

  landmarkToCameraRay(anchorX, anchorY, screenAspect, videoAspect, tanHalfFov, camAspect, _rayAnchor);

  // Unproject ray to 3D metric world position at targetDepth
  _targetPos.copy(_rayAnchor).multiplyScalar(targetDepth);

  // --------------------------------------------------------------------------
  // 4. SCALE — Normalized Biometric Adaptation
  // --------------------------------------------------------------------------
  // Outer canthal distance projected in 3D
  const actualSpan3D = (deltaRayOuter * targetDepth) / cosYaw;
  const rawScale = actualSpan3D / REF_OUTER_CANTHAL_W;
  const targetScale = THREE.MathUtils.clamp(rawScale, 0.85, 1.25);

  // Apply calibrated offsets in head-local coordinate frame
  _targetPos.addScaledVector(_yAxis, Y_OFFSET * targetScale);
  _targetPos.addScaledVector(_zAxis, Z_OFFSET * targetScale);

  // --------------------------------------------------------------------------
  // 5. SMOOTHING — Temporal One-Euro Filter
  // --------------------------------------------------------------------------
  _posFilter.filter(_targetPos, dt, sunglasses.position);
  _quatFilter.filter(_targetQuat, dt, sunglasses.quaternion);
  sunglasses.scale.setScalar(_scaleFilter.filter(targetScale, dt));

  sunglasses.visible = true;
}
