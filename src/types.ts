import type * as THREE from "three";

/**
 * A single face landmark.
 *  - `x`, `y` are in normalized image coordinates: [0, 1] from the top-left.
 *  - `z` is *relative* depth (smaller = closer to the camera), roughly on
 *    the same scale as `x`. It is NOT a metric world coordinate.
 */
export type Landmark = {
  x: number;
  y: number;
  z: number;
};

/**
 * Everything the `update()` function needs from the host scene.
 * Passed in by reference; mutate freely.
 */
export interface UpdateContext {
  /** The sunglasses Group. Pivot is the bridge; +X right, +Y up, +Z toward camera. */
  sunglasses: THREE.Group;
  /** The PerspectiveCamera used to render the scene. */
  camera: THREE.PerspectiveCamera;
  /** The scene, in case you need to add helpers. */
  scene: THREE.Scene;
  /** Current video resolution in pixels. */
  videoWidth: number;
  videoHeight: number;
  /** Seconds elapsed since the previous detected frame. Useful for smoothing. */
  dt: number;
  /**
   * MediaPipe face transformation matrix (4×4, row-major `number[]`).
   * Transforms canonical face model → detected face in camera space.
   * Used for PnP-quality 3D rotation tracking.
   */
  faceMatrix?: number[];
}
