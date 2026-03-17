import type { NextConfig } from "next";

const API_URL = process.env.SMRITIKOSH_API_URL ?? "http://localhost:8080";

const config: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${API_URL}/:path*`,
      },
    ];
  },
};

export default config;
