"use client";

import { useState } from "react";
import { api, type Me } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Modal } from "./Modal";

type Theme = "system" | "light" | "dark";

export function SettingsDialog({ me, onClose, onChanged }: { me: Me; onClose: () => void; onChanged: () => void }) {
  const { authHeaders, user, signOut } = useAuth();
  const [key, setKey] = useState("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The dialog only ever renders in the browser, so reading storage here is safe.
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("tt-theme") as Theme) || "system";
    } catch {
      return "system";
    }
  });

  function applyTheme(t: Theme) {
    setTheme(t);
    try {
      if (t === "system") localStorage.removeItem("tt-theme");
      else localStorage.setItem("tt-theme", t);
    } catch {}
    const root = document.documentElement;
    if (t === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", t);
  }

  async function saveKey(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.saveKey(authHeaders, key.trim());
      setKey("");
      setSaved(true);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title="Settings" onClose={onClose}>
      <form onSubmit={saveKey}>
        <h3 className="font-medium">Gemini API key</h3>
        <p className="mt-1 text-sm text-muted">
          Google Gemini indexes your PDFs and writes the answers with your own key (stored encrypted).{" "}
          <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" className="text-ink underline underline-offset-2">
            Get a free key
          </a>
          . {me.has_api_key ? "A key is saved; paste a new one to replace it." : "No key saved yet."}
        </p>
        <div className="mt-3 flex gap-2">
          <input
            type="password"
            value={key}
            onChange={(e) => {
              setKey(e.target.value);
              setSaved(false);
            }}
            placeholder={me.has_api_key ? "••••••••••••" : "Paste your key"}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-md border border-rule bg-paper px-3 py-2 focus:border-ink focus:outline-none"
          />
          <button disabled={!key.trim() || busy} className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper disabled:opacity-50">
            {busy ? "Saving…" : "Save key"}
          </button>
        </div>
        {saved && <p role="status" className="mt-2 text-sm text-ok">Key saved.</p>}
        {error && <p role="alert" className="mt-2 text-sm text-danger">{error}</p>}
      </form>

      <fieldset className="mt-8">
        <legend className="font-medium">Appearance</legend>
        <div className="mt-2 inline-flex rounded-md border border-rule p-0.5">
          {(["system", "light", "dark"] as Theme[]).map((t) => (
            <label key={t} className={`cursor-pointer rounded px-3 py-1.5 text-sm ${theme === t ? "bg-ink text-paper" : "text-muted hover:text-ink"}`}>
              <input type="radio" name="theme" value={t} checked={theme === t} onChange={() => applyTheme(t)} className="sr-only" />
              {t === "system" ? "Match system" : t === "light" ? "Light" : "Dark"}
            </label>
          ))}
        </div>
      </fieldset>

      <div className="mt-8 flex items-center justify-between border-t border-rule pt-4 text-sm">
        <span className="min-w-0 truncate text-muted">Signed in as {user}</span>
        <button onClick={() => signOut()} className="font-medium hover:text-danger">
          Sign out
        </button>
      </div>
    </Modal>
  );
}
