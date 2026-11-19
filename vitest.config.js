import { defineConfig } from 'vitest/config';

// Separate from vite.config.js, whose root is playground/ (the app).
export default defineConfig({
  test: {
    root: '.',
    include: ['test/**/*.test.js'],
  },
});
