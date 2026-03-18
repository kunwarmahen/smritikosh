import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    accessToken: string;
    user: DefaultSession["user"] & {
      id: string;
      role: "admin" | "user";
      appIds: string[];
    };
  }

  interface User {
    id: string;
    accessToken: string;
    role: "admin" | "user";
    appIds: string[];
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken: string;
    role: "admin" | "user";
    appIds: string[];
  }
}
