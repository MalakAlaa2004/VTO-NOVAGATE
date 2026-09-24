/**
 * ============================================================================
 *  Virtual Try-On (VTO) Tracking Engine — Three.js + MediaPipe FaceLandmarker
 * ============================================================================
 *
 *  Core Responsibilities:
 *  1. Exact Eye-Centered Anchoring:
 *     Computes horizontal anchor as the exact midpoint between the two outer eye
 *     corners (Landmarks 33 and 263), with Y from Landmark 168 (nasion).
 *     Applies calibrated Y_OFFSET (+0.022m) to lift the bridge between the brows
 *     so the lenses frame the eyes squarely and stay well above the nostrils.
 *     Applies Z_OFFSET (+0.015m) forward along the face normal to prevent skin clipping.
 *
 *  2. Scale Factor:
 *     Scaled up by ~20% so the outer rims cleanly match the temporal width.
 *
 *  3. Full 3D Rotation via Orthonormal Basis (Pitch, Yaw, Roll):
 *     X: right-to-left eye vector (263 - 33)
 *     Up: chin (152) to top forehead (10)
 *     Normal Z: X x Up (out of face toward camera)
 *     Orthonormal Y: Z x X
 *
 *  4. Temporal Smoothing (One-Euro Filter):
 *     Eliminates jitter when stationary, instantaneous response during rapid head turns.
 *
 *  5. Head Occlusion Proxy:
 *     Depth-only invisible geometry so temple arms naturally disappear behind the
 *     ears in profile turns.
 *
 *  6. Performance Hygiene:
 *     Zero heap allocations in the hot path. Reusable scratch math hoisted to module scope.
 */

import * as THREE from "three";
import type { Landmark, UpdateContext } from "./types";

// ============================================================================
// MediaPipe Canonical Landmark Indices
// ============================================================================
const LM_NOSE_BRIDGE   = 168; // Nasion / upper nose bridge (ideal glasses anchor)
const LM_FOREHEAD      = 10;  // Top forehead / hairline
const LM_CHIN          = 152; // Chin bottom (full-height pitch leverage)
const LM_RIGHT_EYE_OUT = 33;  // Wearer's right eye outer corner
const LM_LEFT_EYE_OUT  = 263; // Wearer's left eye outer corner

// Biometric reference priors & offsets (metric units: meters)
const REF_BI_CANTHAL_WIDTH = 0.092;  // 92 mm adult outer eye corner distance
const Y_OFFSET             = -0.012; // Shift down directly over the green eye dots & real frames
const Z_OFFSET             = 0.012;  // 12 mm forward offset along face normal Z
const Z_DEPTH_GAIN         = 3.2;    // Calibrates MediaPipe relative depth to true physical pitch & yaw

// ============================================================================
// One-Euro Filter (Casiez et al., CHI 2012)
// Adaptive low-pass filter with velocity-dependent cutoff frequency.
// ============================================================================
class OneEuroFilter1D {
  private xPrev = 0;
  private dxPrev = 0;
  private initialized = false;

  constructor(
    private minCutoff: number = 1.2, // Hz at zero velocity (smooth stationary jitter)
    private beta: number = 0.018,    // Speed coefficient (eliminates lag during motion)
    private dCutoff: number = 1.0    // Derivative filter cutoff in Hz
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
    const alphaD = this.computeAlpha(this.dCutoff, dt);
    const dxHat = alphaD * dx + (1 - alphaD) * this.dxPrev;
    this.dxPrev = dxHat;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const alpha = this.computeAlpha(cutoff, dt);

    const xHat = alpha * x + (1 - alpha) * this.xPrev;
    this.xPrev = xHat;
    return xHat;
  }

  private computeAlpha(cutoff: number, dt: number): number {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }
}

class OneEuroFilterVec3 {
  private fx = new OneEuroFilter1D(1.2, 0.018, 1.0);
  private fy = new OneEuroFilter1D(1.2, 0.018, 1.0);
  private fz = new OneEuroFilter1D(1.2, 0.018, 1.0);

  public reset(): void {
    this.fx.reset();
    this.fy.reset();
    this.fz.reset();
  }

  public filter(target: THREE.Vector3, dt: number, out: THREE.Vector3): THREE.Vector3 {
    out.x = this.fx.filter(target.x, dt);
    out.y = this.fy.filter(target.y, dt);
    out.z = this.fz.filter(target.z, dt);
    return out;
  }
}

class OneEuroFilterQuat {
  private qPrev = new THREE.Quaternion();
  private dOmegaPrev = 0;
  private initialized = false;

  constructor(
    private minCutoff: number = 1.2,
    private beta: number = 0.05,
    private dCutoff: number = 1.0
  ) {}

  public reset(): void {
    this.initialized = false;
  }

  public filter(target: THREE.Quaternion, dt: number, out: THREE.Quaternion): THREE.Quaternion {
    if (!this.initialized || dt <= 0 || dt > 0.5) {
      this.initialized = true;
      this.qPrev.copy(target);
      this.dOmegaPrev = 0;
      out.copy(target);
      return out;
    }

    let dot = this.qPrev.dot(target);
    if (dot < 0) {
      target.x = -target.x;
      target.y = -target.y;
      target.z = -target.z;
      target.w = -target.w;
      dot = -dot;
    }

    const clampedDot = Math.min(Math.max(dot, -1), 1);
    const angle = 2 * Math.acos(clampedDot);
    const omega = angle / dt;

    const tauD = 1.0 / (2.0 * Math.PI * this.dCutoff);
    const alphaD = 1.0 / (1.0 + tauD / dt);
    const omegaHat = alphaD * omega + (1 - alphaD) * this.dOmegaPrev;
    this.dOmegaPrev = omegaHat;

    const cutoff = this.minCutoff + this.beta * omegaHat;
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    const alpha = 1.0 / (1.0 + tau / dt);

    this.qPrev.slerp(target, alpha);
    out.copy(this.qPrev);
    return out;
  }
}

// Module-scope filter instances
const _posFilter = new OneEuroFilterVec3();
const _quatFilter = new OneEuroFilterQuat();
const _scaleFilter = new OneEuroFilter1D(1.0, 0.015, 1.0);

// ============================================================================
// Pre-allocated Scratch Math Objects (Zero Per-Frame Allocations)
// ============================================================================
const _ptRightEye = new THREE.Vector3();
const _ptLeftEye  = new THREE.Vector3();
const _ptForehead = new THREE.Vector3();
const _ptChin     = new THREE.Vector3();
const _ptAnchor   = new THREE.Vector3();

const _xAxis = new THREE.Vector3();
const _yAxis = new THREE.Vector3();
const _zAxis = new THREE.Vector3();
const _vLat  = new THREE.Vector3();
const _vUp   = new THREE.Vector3();

const _basisMatrix = new THREE.Matrix4();
const _targetQuat  = new THREE.Quaternion();
const _targetPos   = new THREE.Vector3();

const _anchorLm: Landmark = { x: 0, y: 0, z: 0 };

// ============================================================================
// Head Occlusion Proxy (Stretch Goal #1)
// Depth-only invisible geometry so temple arms cleanly disappear behind the ears.
// ============================================================================
let _occluderMesh: THREE.Mesh | null = null;

function ensureHeadOccluder(sunglasses: THREE.Group): void {
  if (_occluderMesh) return;

  const headGeo = new THREE.SphereGeometry(1.0, 24, 18);
  // Narrower than temple arms (X = +/-0.048m) so arms stay fully visible from front
  headGeo.scale(0.048, 0.080, 0.070);
  headGeo.translate(0, -0.015, -0.115);

  const occluderMat = new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthWrite: true,
  });

  _occluderMesh = new THREE.Mesh(headGeo, occluderMat);
  _occluderMesh.name = "head_occlusion_proxy";
  _occluderMesh.renderOrder = -1;

  sunglasses.add(_occluderMesh);
}

// ============================================================================
// Coordinate Mapping & Projection
// ============================================================================
function landmarkToCameraRay(
  lm: Landmark,
  screenAspect: number,
  videoAspect: number,
  tanHalfFov: number,
  cameraAspect: number,
  out: THREE.Vector3
): THREE.Vector3 {
  let scaleX = 1.0;
  let scaleY = 1.0;
  if (screenAspect > videoAspect) {
    scaleY = screenAspect / videoAspect;
  } else {
    scaleX = videoAspect / screenAspect;
  }

  const ndcX = (lm.x - 0.5) * 2.0 * scaleX;
  const ndcY = -((lm.y - 0.5) * 2.0 * scaleY);

  out.x = ndcX * tanHalfFov * cameraAspect;
  out.y = ndcY * tanHalfFov;
  out.z = -1.0;
  return out;
}

function landmarkToMetricHeadSpace(
  lm: Landmark,
  anchor: Landmark,
  kMetric: number,
  out: THREE.Vector3
): THREE.Vector3 {
  out.x = (lm.x - anchor.x) * kMetric;
  out.y = -(lm.y - anchor.y) * kMetric;
  out.z = -(lm.z - anchor.z) * kMetric * Z_DEPTH_GAIN;
  return out;
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

  // Extract key facial landmarks
  const lmBridge   = landmarks[LM_NOSE_BRIDGE];   // 168 (Nasion / upper nose bridge)
  const lmForehead = landmarks[LM_FOREHEAD];      // 10  (Top forehead)
  const lmChin     = landmarks[LM_CHIN];          // 152 (Chin bottom)
  const lmRightEye = landmarks[LM_RIGHT_EYE_OUT]; // 33  (Right eye outer)
  const lmLeftEye  = landmarks[LM_LEFT_EYE_OUT];  // 263 (Left eye outer)

  if (!lmBridge || !lmForehead || !lmChin || !lmRightEye || !lmLeftEye) {
    return;
  }

  // Camera projection setup
  const tanHalfFov   = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const cameraAspect = camera.aspect;
  const screenAspect = window.innerWidth / Math.max(window.innerHeight, 1);
  const videoAspect  = videoWidth > 0 && videoHeight > 0 
    ? videoWidth / videoHeight 
    : screenAspect;
  const scaleCompensation = screenAspect > videoAspect ? 1.0 : videoAspect / screenAspect;

  // --------------------------------------------------------------------------
  // 1. Isotropic 3D Head Space & Rigid Orthonormal Basis
  // --------------------------------------------------------------------------
  const dEyeX = lmLeftEye.x - lmRightEye.x;
  const dEyeY = lmLeftEye.y - lmRightEye.y;
  const dEyeZ = lmLeftEye.z - lmRightEye.z;
  const eyeSpanMp = Math.sqrt(dEyeX * dEyeX + dEyeY * dEyeY + dEyeZ * dEyeZ);

  if (eyeSpanMp <= 0.001) {
    return;
  }

  // Metric conversion factor: maps MediaPipe normalized units to meters
  const kMetric = REF_BI_CANTHAL_WIDTH / eyeSpanMp;

  // Transform landmarks to metric 3D head space
  landmarkToMetricHeadSpace(lmRightEye, lmBridge, kMetric, _ptRightEye);
  landmarkToMetricHeadSpace(lmLeftEye,  lmBridge, kMetric, _ptLeftEye);
  landmarkToMetricHeadSpace(lmForehead, lmBridge, kMetric, _ptForehead);
  landmarkToMetricHeadSpace(lmChin,     lmBridge, kMetric, _ptChin);

  // Right vector (X-axis): wearer's right to left eye
  _vLat.subVectors(_ptLeftEye, _ptRightEye);
  _xAxis.copy(_vLat).normalize();

  // Up vector (Y-axis): chin to forehead
  _vUp.subVectors(_ptForehead, _ptChin);

  // Forward vector (Z-axis): pointing out from face
  _zAxis.crossVectors(_xAxis, _vUp).normalize();

  // Re-orthogonalize Y-axis for strict 90-degree basis
  _yAxis.crossVectors(_zAxis, _xAxis).normalize();

  // Apply to rotation matrix
  _basisMatrix.makeBasis(_xAxis, _yAxis, _zAxis);
  _targetQuat.setFromRotationMatrix(_basisMatrix);

  // --------------------------------------------------------------------------
  // 2. Metric Depth & Camera Projection
  // --------------------------------------------------------------------------
  const apparentEyeSpan = eyeSpanMp * 2.0 * scaleCompensation * tanHalfFov * cameraAspect;
  const rawDepth = REF_BI_CANTHAL_WIDTH / Math.max(apparentEyeSpan, 0.001);
  const targetDepth = THREE.MathUtils.clamp(rawDepth, 0.25, 2.50);

  // Precise center between the two outer eye corners horizontally
  // Blend eye-center height and bridge height as recommended
  const eyeMidY = (lmRightEye.y + lmLeftEye.y) * 0.5;
  _anchorLm.x = (lmRightEye.x + lmLeftEye.x) * 0.5;
  _anchorLm.y = (lmBridge.y + eyeMidY) * 0.5;
  _anchorLm.z = (lmRightEye.z + lmLeftEye.z) * 0.5;

  landmarkToCameraRay(_anchorLm, screenAspect, videoAspect, tanHalfFov, cameraAspect, _ptAnchor);
  _targetPos.copy(_ptAnchor).multiplyScalar(targetDepth);

  // --------------------------------------------------------------------------
  // 3. Dynamic Scale (~10% smaller than previous oversized scale) & Offsets
  // --------------------------------------------------------------------------
  const eyeDistance3D = _ptRightEye.distanceTo(_ptLeftEye);
  const rawScale = (eyeDistance3D / REF_BI_CANTHAL_WIDTH) * 1.08;
  const targetScale = THREE.MathUtils.clamp(rawScale, 0.90, 1.25);

  // Apply Y_OFFSET along local Y to align directly over the real frames & eye sockets
  _targetPos.addScaledVector(_yAxis, Y_OFFSET * targetScale);

  // Apply Z_OFFSET along local Z face normal for nose pads clearance
  _targetPos.addScaledVector(_zAxis, Z_OFFSET * targetScale);

  // --------------------------------------------------------------------------
  // 4. Temporal Filtering (One-Euro Filter)
  // --------------------------------------------------------------------------
  _posFilter.filter(_targetPos, dt, sunglasses.position);
  _quatFilter.filter(_targetQuat, dt, sunglasses.quaternion);

  const filteredScale = _scaleFilter.filter(targetScale, dt);
  sunglasses.scale.setScalar(filteredScale);

  sunglasses.visible = true;
}
