// Build-time configuration (NEXT_PUBLIC_* values are inlined into the browser bundle).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

/** Without Supabase configured, the app runs in local dev mode: no passwords, you just
 * pick a workspace name, and the backend must run with AUTH_MODE=dev. */
export const DEV_MODE = !SUPABASE_URL || !SUPABASE_ANON_KEY;
