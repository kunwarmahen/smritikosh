"use client";
import { useEffect, useState } from "react";

const KEY = "smriti-theme";

export function useTheme() {
  const [dark, setDark] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem(KEY);
    const isDark = stored !== "light";
    setDark(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  const toggle = () => {
    setDark((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle("dark", next);
      localStorage.setItem(KEY, next ? "dark" : "light");
      return next;
    });
  };

  return { dark, toggle };
}
