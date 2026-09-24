import * as THREE from "three";
import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import type { FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import { createSunglasses } from "./sunglasses";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { update } from "./update";
import type { Landmark, UpdateContext } from "./types";
 
// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
function $required<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`index.html is missing #${id} — re-extract the project`);
  }
  return el as T;
}
 
const video       = $required<HTMLVideoElement>("video");
const canvas      = $required<HTMLCanvasElement>("canvas");
// HUD elements are non-essential; guard their use.
const fpsEl       = document.getElementById("fps")         as HTMLElement | null;
const faceDot     = document.getElementById("face-dot")    as HTMLElement | null;
const faceStatus  = document.getElementById("face-status") as HTMLElement | null;
const videoSizeEl = document.getElementById("video-size")  as HTMLElement | null;
const errorBanner = document.getElementById("error")       as HTMLElement | null;
 
function showError(message: string): void {
  console.error("[ar-vto]", message);
  if (errorBanner) {
    errorBanner.textContent = message;
    errorBanner.classList.add("on");
  } else {
    // index.html wasn't updated with #error — fall back to a plain alert
    // so the candidate still gets a clear signal.
    alert(message);
  }
  if (faceStatus) faceStatus.textContent = "error";
  faceDot?.classList.remove("on");
}
 
// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
 
// 45° vertical FOV is a reasonable starting guess for a typical laptop
// webcam. Tweak as you see fit — it directly affects how landmark x/y
// rays project into world space.
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.01,
  100,
);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);
 
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: true,
  premultipliedAlpha: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
 
const key = new THREE.DirectionalLight(0xffffff, 1.0);
key.position.set(0.5, 1, 0.8);
scene.add(key);
 
const rim = new THREE.DirectionalLight(0x88aaff, 0.4);
rim.position.set(-0.5, 0.2, -0.5);
scene.add(rim);
 
// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
const sunglasses = createSunglasses();
// Start it visible somewhere so you can confirm it loaded on first run.
sunglasses.position.set(0, 0, -0.5);
scene.add(sunglasses);
 
// ---------------------------------------------------------------------------
// Optional landmark debug overlay (toggle with D)
// ---------------------------------------------------------------------------
const NUM_LANDMARKS = 478; // MediaPipe FaceLandmarker returns 478 points
const debugGeo = new THREE.BufferGeometry();
debugGeo.setAttribute(
  "position",
  new THREE.BufferAttribute(new Float32Array(NUM_LANDMARKS * 3), 3),
);
const debugPoints = new THREE.Points(
  debugGeo,
  new THREE.PointsMaterial({ color: 0x4ade80, size: 0.003, sizeAttenuation: true }),
);
debugPoints.visible = false;
debugPoints.frustumCulled = false;
scene.add(debugPoints);
 
let debugEnabled = false;
window.addEventListener("keydown", (e) => {
  if (e.key === "d" || e.key === "D") {
    debugEnabled = !debugEnabled;
    debugPoints.visible = debugEnabled;
  }
});
 
// Helper for debug visuals only. Unprojects a normalized landmark to a
// world-space point on a plane at distance `depth` in front of the camera.
const _tmpV = new THREE.Vector3();
const _tmpDir = new THREE.Vector3();
function landmarkToWorldDebug(lm: Landmark, depth = 0.5): THREE.Vector3 {
  const ndcX = lm.x * 2 - 1;
  const ndcY = -(lm.y * 2 - 1); // image y goes down, NDC y goes up
  _tmpV.set(ndcX, ndcY, 0.5).unproject(camera);
  _tmpDir.copy(_tmpV).sub(camera.position).normalize();
  const t = depth / -_tmpDir.z;
  return _tmpV.copy(camera.position).addScaledVector(_tmpDir, t);
}
 
function updateDebugPoints(landmarks: Landmark[]): void {
  const arr = debugGeo.attributes.position.array as Float32Array;
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarkToWorldDebug(landmarks[i]!, 0.5);
    arr[i * 3 + 0] = p.x;
    arr[i * 3 + 1] = p.y;
    arr[i * 3 + 2] = p.z;
  }
  debugGeo.attributes.position.needsUpdate = true;
}
 
// ---------------------------------------------------------------------------
// Webcam
//
// We try a chain of progressively looser constraints. Many desktop webcams
// (especially on Windows) don't advertise a `facingMode`, so requiring
// `"user"` makes them appear nonexistent — hence the NotFoundError some
// candidates see. By the time we hit `{ video: true }` any working camera
// the OS exposes will be accepted.
// ---------------------------------------------------------------------------
const CAMERA_ATTEMPTS: MediaStreamConstraints[] = [
  { video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
  { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
  { video: true, audio: false },
];
 
async function startCamera(): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("getUserMedia not available — open the page over HTTPS or http://localhost");
  }
 
  // Sanity-check what the OS exposes BEFORE asking for a stream. This lets
  // us tell "no cameras present at OS level" apart from "constraints too
  // strict for the cameras that ARE present."
  let cameraDeviceCount = -1;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    cameraDeviceCount = devices.filter((d) => d.kind === "videoinput").length;
    console.info(`[ar-vto] ${cameraDeviceCount} video input device(s) reported by OS.`);
  } catch (err) {
    console.warn("[ar-vto] enumerateDevices failed:", err);
  }
 
  let lastErr: unknown;
  for (const constraints of CAMERA_ATTEMPTS) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      await new Promise<void>((res) =>
        video.addEventListener("loadedmetadata", () => res(), { once: true }),
      );
      await video.play();
      if (videoSizeEl) videoSizeEl.textContent = `${video.videoWidth}×${video.videoHeight}`;
      resize();
      return;
    } catch (err) {
      lastErr = err;
      // Try the next set of constraints.
    }
  }
 
  // All attempts failed. If we know zero cameras were enumerated, surface
  // a more actionable message than the generic NotFoundError.
  if (cameraDeviceCount === 0) {
    throw new DOMException(
      "Your OS is reporting zero cameras to the browser. " +
        "On Windows: Settings → Privacy & security → Camera — enable both " +
        '"Camera access" and "Let apps access your camera", then reload. ' +
        "Also confirm no other app (Teams, Zoom, OBS) is holding the camera, " +
        "and that any privacy-shutter / hardware kill-switch is open.",
      "NotFoundError",
    );
  }
  throw lastErr;
}
 
/**
 * Turn a getUserMedia DOMException into something the candidate can act on.
 */
function describeMediaError(err: unknown): string {
  if (!(err instanceof DOMException)) {
    return err instanceof Error ? err.message : String(err);
  }
  switch (err.name) {
    case "NotFoundError":
    case "DevicesNotFoundError":
      // If we constructed our own descriptive message above, surface it.
      return err.message && err.message.length > 40
        ? err.message
        : "No camera detected. Check that one is connected and enabled in OS privacy settings.";
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera permission denied. Allow it in the browser's site settings and reload.";
    case "NotReadableError":
    case "TrackStartError":
      return "Camera is in use by another application. Close anything else using it and reload.";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return `Camera doesn't match constraints (${err.message || "unknown"}).`;
    case "SecurityError":
      return "Page must be served over HTTPS or http://localhost.";
    default:
      return `${err.name}: ${err.message}`;
  }
}
 
function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
 
// ---------------------------------------------------------------------------
// MediaPipe FaceLandmarker
// ---------------------------------------------------------------------------
let faceLandmarker: FaceLandmarker | null = null;
 
async function startFaceLandmarker(): Promise<void> {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm",
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}
 
// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
let lastVideoTime = -1;
let lastDetectMs  = -1;
 
let fpsFrames = 0;
let fpsTimer  = performance.now();
 
// Reusable context object — single allocation, mutated each frame.
const ctx: UpdateContext = {
  sunglasses,
  camera,
  scene,
  videoWidth: 0,
  videoHeight: 0,
  dt: 0,
};
 
function loop(): void {
  requestAnimationFrame(loop);
 
  if (faceLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
 
    const nowMs = performance.now();
    const result: FaceLandmarkerResult = faceLandmarker.detectForVideo(video, nowMs);
 
    const face = result.faceLandmarks?.[0];
    if (face) {
      faceDot?.classList.add("on");
      if (faceStatus) faceStatus.textContent = "detected";
 
      if (debugEnabled) updateDebugPoints(face);
 
      ctx.videoWidth  = video.videoWidth;
      ctx.videoHeight = video.videoHeight;
      ctx.dt = lastDetectMs < 0 ? 0 : (nowMs - lastDetectMs) / 1000;
      lastDetectMs = nowMs;
 
      // ---- candidate code runs here ----
      update(face, ctx);
      // ----------------------------------
    } else {
      faceDot?.classList.remove("on");
      if (faceStatus) faceStatus.textContent = "searching…";
      lastDetectMs = -1;
    }
  }
 
  renderer.render(scene, camera);
 
  // FPS readout
  fpsFrames++;
  const now = performance.now();
  if (now - fpsTimer >= 500) {
    if (fpsEl) fpsEl.textContent = ((fpsFrames * 1000) / (now - fpsTimer)).toFixed(0);
    fpsFrames = 0;
    fpsTimer = now;
  }
}
 
// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async () => {
  try {
    await startCamera();
    await startFaceLandmarker();
    loop();
  } catch (err) {
    console.error(err);
    showError(describeMediaError(err));
  }
})();