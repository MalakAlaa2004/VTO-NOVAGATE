import { defineConfig, createLogger, type LogType } from "vite";


const logger = createLogger();
const isMediaPipeSourcemapNoise = (msg: string) =>
  typeof msg === "string" &&
  msg.includes("Failed to load source map") &&
  msg.includes("vision_bundle");
 
(["info", "warn", "warnOnce", "error"] as LogType[]).forEach((level) => {
  const original = logger[level].bind(logger);
  logger[level] = (msg, opts) => {
    if (isMediaPipeSourcemapNoise(msg)) return;
    original(msg, opts);
  };
});
 
export default defineConfig({
  customLogger: logger,
  server: {
    host: true,
    port: 5173,
    allowedHosts: true,
  },
  optimizeDeps: {
    // MediaPipe loads its wasm at runtime; pre-bundling breaks it.
    exclude: ["@mediapipe/tasks-vision"],
  },
  
});
 