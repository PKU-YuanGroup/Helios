import { defineConfig } from "vite";

export default defineConfig({
  // Keep relative Wasm imports intact in dev; Vite bundles SDK 3.3's embedded
  // Wasm JS automatically for production. Prebundle CommonJS dependencies.
  optimizeDeps: {
    exclude: ["@reactor-team/js-sdk"],
    include: ["awaitqueue", "react", "react/jsx-runtime", "hls.js", "mp4box"],
  },
});
