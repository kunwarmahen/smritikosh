import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    accessToken: string;
    user: DefaultSession["user"] & {
      id: string;
      role: "admin" | "user";
      appId: string;
    };
  }

  interface User {
    id: string;
    accessToken: string;
    role: "admin" | "user";
    appId: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken: string;
    role: "admin" | "user";
    appId: string;
  }
}
