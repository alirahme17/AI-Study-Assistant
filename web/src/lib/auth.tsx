"use client";

import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { DEV_MODE, SUPABASE_ANON_KEY, SUPABASE_URL } from "./config";

const DEV_USER_KEY = "tt-dev-user";

let supabase: SupabaseClient | null = null;
function getSupabase(): SupabaseClient | null {
  if (DEV_MODE) return null;
  if (!supabase) supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  return supabase;
}

type AuthState = {
  ready: boolean;
  /** Email (Supabase) or workspace name (dev mode); null when signed out. */
  user: string | null;
  /** Headers that identify the user to the backend. */
  authHeaders: () => Promise<Record<string, string>>;
  signIn: (email: string, password: string) => Promise<string | null>;
  signUp: (email: string, password: string) => Promise<string | null>;
  devSignIn: (workspace: string) => void;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

// Dev-mode workspace name in localStorage, read as an external store (no hydration mismatch).
const DEV_EVENT = "tt-dev-user-change";
function subscribeDevUser(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(DEV_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(DEV_EVENT, callback);
  };
}
function readDevUser(): string | null {
  try {
    return localStorage.getItem(DEV_USER_KEY);
  } catch {
    return null;
  }
}
function writeDevUser(value: string | null) {
  try {
    if (value) localStorage.setItem(DEV_USER_KEY, value);
    else localStorage.removeItem(DEV_USER_KEY);
  } catch {
    /* storage unavailable: the session just won't survive a reload */
  }
  window.dispatchEvent(new Event(DEV_EVENT));
}
const noopSubscribe = () => () => {};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // false during server rendering and hydration, true afterwards
  const hydrated = useSyncExternalStore(noopSubscribe, () => true, () => false);
  const devUser = useSyncExternalStore(subscribeDevUser, readDevUser, () => null);
  const [session, setSession] = useState<Session | null>(null);
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const ready = DEV_MODE ? hydrated : sessionLoaded;

  useEffect(() => {
    const client = getSupabase();
    if (!client) return;
    client.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setSessionLoaded(true);
    });
    const { data } = client.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, []);

  const authHeaders = useCallback(async (): Promise<Record<string, string>> => {
    const client = getSupabase();
    if (!client) return devUser ? { "X-Dev-User": devUser } : {};
    const { data } = await client.auth.getSession(); // refreshes an expired token
    return data.session ? { Authorization: `Bearer ${data.session.access_token}` } : {};
  }, [devUser]);

  const value = useMemo<AuthState>(
    () => ({
      ready,
      user: DEV_MODE ? devUser : session?.user.email ?? null,
      authHeaders,
      async signIn(email, password) {
        const { error } = await getSupabase()!.auth.signInWithPassword({ email, password });
        return error ? error.message : null;
      },
      async signUp(email, password) {
        const { error } = await getSupabase()!.auth.signUp({ email, password });
        return error ? error.message : null;
      },
      devSignIn(workspace) {
        writeDevUser(workspace);
      },
      async signOut() {
        if (DEV_MODE) {
          writeDevUser(null);
        } else {
          await getSupabase()!.auth.signOut();
        }
      },
    }),
    [ready, devUser, session, authHeaders],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
