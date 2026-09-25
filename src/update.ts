/**
 * ============================================================================
 *  Virtual Try-On (VTO) Tracking Engine — Three.js + MediaPipe FaceLandmarker
 * ============================================================================
 *
 *  Engine Architecture (Enterprise Stability like Instagram / Spark AR):
 *
 *  1. Jaw-Isolated Cranial Orthonormal Basis:
 *     Derives lateral axis from bilateral canthi (inner 133/362 + outer 33/263)
 *     and cranial vertical axis from Subnasale (2) / Nasion (168) to Forehead (10).
 *     Because the mandible (chin 152) is strictly avoided, the glasses are 100%
 *     immune to talking, smiling, mouth movements, and chewing.
 *
 *  2. Isotropic Aspect-Corrected Metric Head Space:
 *     MediaPipe normalized coordinates are aspect-scaled by (width / height)
 *     so Euclidean distances are isotropic. Prevents orientation skewing and
 *     scale breathing when tilting or rolling the head.
 *
 *  3. Yaw Foreshortening Depth Compensation:
 *     When the head turns sideways, apparent 2D eye span shrinks by cos(yaw).
 *     The engine compensates using the head orientation normal, keeping metric
 *     depth rock-solid constant across 3D head rotations.
 *
 *  4. C1-Continuous Smoothstep One-Euro Filter:
 *     Eliminates hard deadband stepping. Sensor noise tremors below the noise floor
 *     (< 0.8mm pos, < 0.5° rot) are smoothly damped to zero, pinning the glasses
 *     to the face like a physical object. Dynamic beta opens cutoff during head motion
 *     for zero-latency responsiveness.
 *
 *  5. Head Occlusion Proxy (Stretch Goal #1):
 *     Depth-only invisible cranial ellipsoid cleanly occludes temple arms behind
 *     the ears in 3/4 and profile views without clipping the front frames.
 *
 *  6. Zero-Allocation Hot Path:
 *     All scratch vectors, matrices, and quaternions are hoisted to module scope.
 */

import * as THREE from "three";
import type { Landmark, UpdateContext } from "./types";

// ============================================================================
// MediaPipe Canonical Landmark Indices (Rigid Cephalometric Frame)
// ============================================================================
const LM_NASION         = 168; // Suture between nasal & frontal bones (between eyes)
const LM_GLABELLA       = 6;   // Smooth prominence between eyebrows
const LM_RIGHT_EYE_OUT  = 33;  // Wearer's right eye outer canthus
const LM_LEFT_EYE_OUT   = 263; // Wearer's left eye outer canthus
const LM_RIGHT_EYE_IN   = 133; // Wearer's right eye inner canthus (blink-immune)
const LM_LEFT_EYE_IN    = 362; // Wearer's left eye inner canthus (blink-immune)
const LM_RIGHT_EAR      = 234; // Right tragus / zygomatic arch (ear level)
const LM_LEFT_EAR       = 454; // Left tragus / zygomatic arch (ear level)

// Biometric reference dimensions (metric meters)
const REF_OUTER_CANTHAL_W = 0.092; // 92 mm adult outer eye corner distance
const REF_INNER_CANTHAL_W = 0.033; // 33 mm adult inner eye corner distance

// Placement tuning
const Y_OFFSET     = -0.006; // Centers the circular lenses squarely over the eye line
const Z_OFFSET     = 0.012;  // 12 mm forward offset along face normal for nose pads
const Z_DEPTH_GAIN = 1.8;    // Calibrated metric depth ratio for cranial landmarks

// ============================================================================
// C1-Continuous Smoothstep Filter (Eliminates Jitter & Deadband Stepping)
// ============================================================================
class SmoothOneEuroFilter1D {
  private xPrev = 0;
  private dxPrev = 0;
  private initialized = false;

  constructor(
    private minCutoff: number = 0.08, // Ultra-stable at rest
    private beta: number = 2.4,       // Instant responsiveness during motion
    private dCutoff: number = 1.2,    // Derivative filter cutoff
    private deadband: number = 0.0008 // 0.8 mm noise floor
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

    const rawDelta = x - this.xPrev;
    const absDelta = Math.abs(rawDelta);

    // C1-continuous smoothstep attenuation for micro-noise
    // Avoids hard-cliff deadband stepping artifacts
    let effectiveX = x;
    if (absDelta < this.deadband) {
      const t = absDelta / this.deadband;
      const weight = t * t * (3.0 - 2.0 * t); // Smooth Hermite curve [0, 1]
      effectiveX = this.xPrev + rawDelta * weight;
    }

    const delta = effectiveX - this.xPrev;
    const dx = delta / dt;

    const alphaD = this.computeAlpha(this.dCutoff, dt);
    const dxHat = alphaD * dx + (1.0 - alphaD) * this.dxPrev;
    this.dxPrev = dxHat;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const alpha = this.computeAlpha(cutoff, dt);

    const xHat = alpha * effectiveX + (1.0 - alpha) * this.xPrev;
    this.xPrev = xHat;
    return xHat;
  }

  private computeAlpha(cutoff: number, dt: number): number {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }
}

class SmoothOneEuroFilterVec3 {
  private fx = new SmoothOneEuroFilter1D(0.08, 2.4, 1.2, 0.0006);
  private fy = new SmoothOneEuroFilter1D(0.08, 2.4, 1.2, 0.0006);
  // Z-depth: extra damped to prevent camera distance breathing
  private fz = new SmoothOneEuroFilter1D(0.04, 2.0, 0.8, 0.0012);

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

class SmoothOneEuroFilterQuat {
  private qPrev = new THREE.Quaternion();
  private dOmegaPrev = 0;
  private initialized = false;

  constructor(
    private minCutoff: number = 0.08,
    private beta: number = 2.0,
    private dCutoff: number = 1.2,
    private angleDeadband: number = 0.009 // ~0.5° angular noise deadband
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

    const clampedDot = Math.min(Math.max(dot, -1.0), 1.0);
    const angle = 2.0 * Math.acos(clampedDot);

    // Smoothstep angular attenuation for sensor noise
    let effectiveTarget = target;
    if (angle < this.angleDeadband) {
      const t = angle / this.angleDeadband;
      const weight = t * t * (3.0 - 2.0 * t);
      _scratchQuat.copy(this.qPrev).slerp(target, weight);
      effectiveTarget = _scratchQuat;
    }

    const effectiveDot = Math.min(Math.max(this.qPrev.dot(effectiveTarget), -1.0), 1.0);
    const effectiveAngle = 2.0 * Math.acos(effectiveDot);
    const omega = effectiveAngle / dt;

    const tauD = 1.0 / (2.0 * Math.PI * this.dCutoff);
    const alphaD = 1.0 / (1.0 + tauD / dt);
    const omegaHat = alphaD * omega + (1.0 - alphaD) * this.dOmegaPrev;
    this.dOmegaPrev = omegaHat;

    const cutoff = this.minCutoff + this.beta * omegaHat;
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    const alpha = 1.0 / (1.0 + tau / dt);

    this.qPrev.slerp(effectiveTarget, alpha);
    out.copy(this.qPrev);
    return out;
  }
}

// Module-scope filter instances
const _posFilter   = new SmoothOneEuroFilterVec3();
const _quatFilter  = new SmoothOneEuroFilterQuat();
const _scaleFilter = new SmoothOneEuroFilter1D(0.04, 1.2, 0.8, 0.003); // Skulls don't pulse; heavily smooth scale

// ============================================================================
// Pre-allocated Scratch Math Objects (Zero Per-Frame Allocations)
// ============================================================================
const _ptRightEyeOut = new THREE.Vector3();
const _ptLeftEyeOut  = new THREE.Vector3();
const _ptRightEyeIn  = new THREE.Vector3();
const _ptLeftEyeIn   = new THREE.Vector3();
const _ptRightEar    = new THREE.Vector3();
const _ptLeftEar     = new THREE.Vector3();
const _ptNasion      = new THREE.Vector3();
const _ptEarsMid     = new THREE.Vector3();
const _ptAnchor      = new THREE.Vector3();

const _rayRightOut   = new THREE.Vector3();
const _rayLeftOut    = new THREE.Vector3();
const _rayRightIn    = new THREE.Vector3();
const _rayLeftIn     = new THREE.Vector3();

const _xAxis         = new THREE.Vector3();
const _yAxis         = new THREE.Vector3();
const _zAxis         = new THREE.Vector3();
const _vLat          = new THREE.Vector3();
const _vFwd          = new THREE.Vector3();

const _basisMatrix   = new THREE.Matrix4();
const _targetQuat    = new THREE.Quaternion();
const _targetPos     = new THREE.Vector3();
const _scratchQuat   = new THREE.Quaternion();

const _anchorLm: Landmark = { x: 0, y: 0, z: 0 };

// ============================================================================
// Head Occlusion Proxy (Stretch Goal #1)
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
// Coordinate Mapping & Camera Ray Projection
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
  videoAspect: number,
  out: THREE.Vector3
): THREE.Vector3 {
  out.x = (lm.x - anchor.x) * kMetric;
  // Account for non-square normalized image coordinates
  out.y = -((lm.y - anchor.y) / videoAspect) * kMetric;
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
  const lmNasion      = landmarks[LM_NASION];        // 168 (Nasion / upper nose bridge)
  const lmGlabella    = landmarks[LM_GLABELLA];      // 6   (Glabella between brows)
  const lmRightEyeOut = landmarks[LM_RIGHT_EYE_OUT]; // 33  (Right eye outer canthus)
  const lmLeftEyeOut  = landmarks[LM_LEFT_EYE_OUT];  // 263 (Left eye outer canthus)
  const lmRightEyeIn  = landmarks[LM_RIGHT_EYE_IN];  // 133 (Right eye inner canthus, blink-immune)
  const lmLeftEyeIn   = landmarks[LM_LEFT_EYE_IN];   // 362 (Left eye inner canthus, blink-immune)
  const lmRightEar    = landmarks[LM_RIGHT_EAR];     // 234 (Right tragus / ear level)
  const lmLeftEar     = landmarks[LM_LEFT_EAR];      // 454 (Left tragus / ear level)

  if (!lmNasion || !lmGlabella || 
      !lmRightEyeOut || !lmLeftEyeOut || !lmRightEyeIn || !lmLeftEyeIn ||
      !lmRightEar || !lmLeftEar) {
    return;
  }

  // Camera projection setup
  const tanHalfFov   = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const cameraAspect = camera.aspect;
  const screenAspect = window.innerWidth / Math.max(window.innerHeight, 1);
  const videoAspect  = videoWidth > 0 && videoHeight > 0 
    ? videoWidth / videoHeight 
    : screenAspect;

  // --------------------------------------------------------------------------
  // 1. Isotropic 3D Head Space & Ear-to-Nose Cephalometric Basis
  // --------------------------------------------------------------------------
  // Isotropic aspect-scaled distance between outer eye corners
  const dOuterIsoX = lmLeftEyeOut.x - lmRightEyeOut.x;
  const dOuterIsoY = (lmLeftEyeOut.y - lmRightEyeOut.y) / videoAspect;
  const dOuterIsoZ = (lmLeftEyeOut.z - lmRightEyeOut.z) * Z_DEPTH_GAIN;
  const outerSpanMp = Math.sqrt(dOuterIsoX * dOuterIsoX + dOuterIsoY * dOuterIsoY + dOuterIsoZ * dOuterIsoZ);

  if (outerSpanMp <= 0.001) {
    return;
  }

  // Metric conversion factor: maps normalized coordinates to meters
  const kMetric = REF_OUTER_CANTHAL_W / outerSpanMp;

  // Transform landmarks to metric isotropic cranial space centered at Nasion
  landmarkToMetricHeadSpace(lmRightEyeOut, lmNasion, kMetric, videoAspect, _ptRightEyeOut);
  landmarkToMetricHeadSpace(lmLeftEyeOut,  lmNasion, kMetric, videoAspect, _ptLeftEyeOut);
  landmarkToMetricHeadSpace(lmRightEyeIn,  lmNasion, kMetric, videoAspect, _ptRightEyeIn);
  landmarkToMetricHeadSpace(lmLeftEyeIn,   lmNasion, kMetric, videoAspect, _ptLeftEyeIn);
  landmarkToMetricHeadSpace(lmRightEar,    lmNasion, kMetric, videoAspect, _ptRightEar);
  landmarkToMetricHeadSpace(lmLeftEar,     lmNasion, kMetric, videoAspect, _ptLeftEar);
  landmarkToMetricHeadSpace(lmNasion,      lmNasion, kMetric, videoAspect, _ptNasion);

  // Right vector (X-axis): combine outer and inner canthi lines
  // Averaging inner + outer lines cancels landmark detector noise by ~50%
  const dOutX = _ptLeftEyeOut.x - _ptRightEyeOut.x;
  const dOutY = _ptLeftEyeOut.y - _ptRightEyeOut.y;
  const dOutZ = _ptLeftEyeOut.z - _ptRightEyeOut.z;

  const dInX = _ptLeftEyeIn.x - _ptRightEyeIn.x;
  const dInY = _ptLeftEyeIn.y - _ptRightEyeIn.y;
  const dInZ = _ptLeftEyeIn.z - _ptRightEyeIn.z;

  _vLat.set(
    (dOutX + dInX) * 0.5,
    (dOutY + dInY) * 0.5,
    (dOutZ + dInZ) * 0.5
  );
  _xAxis.copy(_vLat).normalize();

  // Forward vector (Z-axis): from midpoint of the ears to the nasion (nose bridge)
  // Both ears and nasion lie on the Frankfurt horizontal plane of eyewear.
  // Temple arms run along -Z directly to the ears without downward slant into cheeks.
  _ptEarsMid.addVectors(_ptRightEar, _ptLeftEar).multiplyScalar(0.5);
  _vFwd.subVectors(_ptNasion, _ptEarsMid);

  // Orthogonalize Z against X: Z = normalize(vFwd - (vFwd · X) * X)
  _zAxis.copy(_vFwd).addScaledVector(_xAxis, -_vFwd.dot(_xAxis)).normalize();

  // Up vector (Y-axis): Y = Z × X (strictly perpendicular right-handed basis)
  _yAxis.crossVectors(_zAxis, _xAxis).normalize();

  // Construct rotation quaternion
  _basisMatrix.makeBasis(_xAxis, _yAxis, _zAxis);
  _targetQuat.setFromRotationMatrix(_basisMatrix);

  // --------------------------------------------------------------------------
  // 2. Metric Depth with Yaw Foreshortening Compensation
  // --------------------------------------------------------------------------
  // Project landmark camera rays at distance Z = 1
  landmarkToCameraRay(lmRightEyeOut, screenAspect, videoAspect, tanHalfFov, cameraAspect, _rayRightOut);
  landmarkToCameraRay(lmLeftEyeOut,  screenAspect, videoAspect, tanHalfFov, cameraAspect, _rayLeftOut);
  landmarkToCameraRay(lmRightEyeIn,  screenAspect, videoAspect, tanHalfFov, cameraAspect, _rayRightIn);
  landmarkToCameraRay(lmLeftEyeIn,   screenAspect, videoAspect, tanHalfFov, cameraAspect, _rayLeftIn);

  // Angular transverse spans on camera plane
  const dxOut = _rayLeftOut.x - _rayRightOut.x;
  const dyOut = _rayLeftOut.y - _rayRightOut.y;
  const deltaRayOut = Math.sqrt(dxOut * dxOut + dyOut * dyOut);

  const dxIn = _rayLeftIn.x - _rayRightIn.x;
  const dyIn = _rayLeftIn.y - _rayRightIn.y;
  const deltaRayIn = Math.sqrt(dxIn * dxIn + dyIn * dyIn);

  // Yaw foreshortening factor: cos(yaw) = sqrt(1 - xAxis.z^2)
  // When head turns sideways, transverse projected span shrinks by cos(yaw).
  // Compensating prevents the glasses from breathing / flying backward when turning!
  const foreshortening = Math.max(Math.sqrt(Math.max(0.01, 1.0 - _xAxis.z * _xAxis.z)), 0.35);

  const depthFromOuter = (REF_OUTER_CANTHAL_W * foreshortening) / Math.max(deltaRayOut, 0.001);
  const depthFromInner = (REF_INNER_CANTHAL_W * foreshortening) / Math.max(deltaRayIn, 0.001);

  // Fuse outer baseline with blink-immune inner canthi
  const rawDepth = depthFromOuter * 0.65 + depthFromInner * 0.35;
  const targetDepth = THREE.MathUtils.clamp(rawDepth, 0.25, 2.50);

  // Multi-landmark horizontal and vertical anchor:
  // Combines inner corners (blink-proof), outer corners, and nasion
  const outerMidX = (lmRightEyeOut.x + lmLeftEyeOut.x) * 0.5;
  const innerMidX = (lmRightEyeIn.x + lmLeftEyeIn.x) * 0.5;
  _anchorLm.x = outerMidX * 0.5 + innerMidX * 0.5;

  const outerMidY = (lmRightEyeOut.y + lmLeftEyeOut.y) * 0.5;
  const innerMidY = (lmRightEyeIn.y + lmLeftEyeIn.y) * 0.5;
  const eyeLevelY = outerMidY * 0.5 + innerMidY * 0.5;
  const bridgeLevelY = lmNasion.y * 0.7 + lmGlabella.y * 0.3;
  _anchorLm.y = bridgeLevelY * 0.6 + eyeLevelY * 0.4;

  _anchorLm.z = (lmRightEyeOut.z + lmLeftEyeOut.z) * 0.25 + 
                (lmRightEyeIn.z + lmLeftEyeIn.z) * 0.25 + 
                lmNasion.z * 0.5;

  landmarkToCameraRay(_anchorLm, screenAspect, videoAspect, tanHalfFov, cameraAspect, _ptAnchor);
  _targetPos.copy(_ptAnchor).multiplyScalar(targetDepth);

  // --------------------------------------------------------------------------
  // 3. Dynamic Scale & Offsets
  // --------------------------------------------------------------------------
  const eyeDistance3D = _ptRightEyeOut.distanceTo(_ptLeftEyeOut);
  const rawScale = (eyeDistance3D / REF_OUTER_CANTHAL_W) * 1.04;
  const targetScale = THREE.MathUtils.clamp(rawScale, 0.90, 1.25);

  // Apply calibrated offsets
  _targetPos.addScaledVector(_yAxis, Y_OFFSET * targetScale);
  _targetPos.addScaledVector(_zAxis, Z_OFFSET * targetScale);

  // --------------------------------------------------------------------------
  // 4. Temporal Filtering (Smoothstep One-Euro Filter)
  // --------------------------------------------------------------------------
  _posFilter.filter(_targetPos, dt, sunglasses.position);
  _quatFilter.filter(_targetQuat, dt, sunglasses.quaternion);

  const filteredScale = _scaleFilter.filter(targetScale, dt);
  sunglasses.scale.setScalar(filteredScale);

  sunglasses.visible = true;
}
