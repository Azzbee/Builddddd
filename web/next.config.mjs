/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // `next build` and `next dev` share .next by default, so a build run while the
  // dev server is up corrupts its chunks ("Cannot find module './NNN.js'").
  // Verification builds use `npm run build:check` (a separate dist dir) instead.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  outputFileTracingRoot: process.cwd(),
};
export default nextConfig;
