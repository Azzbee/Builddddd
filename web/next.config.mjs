/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // `next build` and `next dev` share .next by default, so a build run while the
  // dev server is up corrupts its chunks ("Cannot find module './NNN.js'").
  // Verification builds use `npm run build:check` (a separate dist dir) instead.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  async rewrites() {
    // Proxy API calls to the backend in dev so the browser hits same-origin.
    const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${base}/:path*` }];
  },
};
export default nextConfig;
