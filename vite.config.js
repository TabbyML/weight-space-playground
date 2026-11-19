import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  // Vite root is the playground; the build lands at the repo-root dist/.
  root: fileURLToPath(new URL('./playground', import.meta.url)),
  // Relative asset URLs so the built site works from any origin root.
  base: './',
  // Files copied verbatim into dist/ (notably data/).
  publicDir: 'public',
  // Keep the dep cache at the repo root instead of under playground/.
  cacheDir: fileURLToPath(new URL('./node_modules/.vite', import.meta.url)),
  server: {
    // Cross-origin isolation for local dev so wllama's multi-threaded WASM can
    // use SharedArrayBuffer. GitHub Pages can't set these headers, so the
    // deployed site runs the single-threaded fallback.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    // Outside the Vite root, so emptyOutDir must be explicit.
    outDir: fileURLToPath(new URL('./dist', import.meta.url)),
    emptyOutDir: true,
  },
});
