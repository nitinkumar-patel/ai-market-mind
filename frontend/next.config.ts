import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Produces a self-contained server.js used by the production Docker image.
  // In dev (npm run dev) this has no effect.
  output: 'standalone',

  // Proxy /api/* to the FastAPI backend — avoids CORS in local dev.
  // In production, nginx handles this routing before requests reach Next.js.
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_BACKEND_URL ?? 'http://localhost:8000'}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
