/** @type {import('next').NextConfig} */
const nextConfig = {
  // Note: Removed "output: export" to enable API routes for Turso database
  trailingSlash: true,
  images: { unoptimized: true }
};

export default nextConfig;

