"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { DEV_MODE } from "@/lib/config";

export default function LoginPage() {
  const { ready, user, signIn, signUp, devSignIn } = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [workspace, setWorkspace] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (ready && user) router.replace("/");
  }, [ready, user, router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    if (DEV_MODE) {
      if (!/^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}$/.test(workspace.trim())) {
        setError("Use letters, numbers, dashes or underscores.");
        return;
      }
      devSignIn(workspace.trim());
      return;
    }
    setBusy(true);
    const err = mode === "signin" ? await signIn(email, password) : await signUp(email, password);
    setBusy(false);
    if (err) setError(err);
    else if (mode === "signup") setNotice("Check your inbox to confirm your email, then sign in.");
  }

  return (
    <main className="min-h-full grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <section className="flex flex-col justify-center px-6 py-12 sm:px-12 lg:px-20">
        <div className="w-full max-w-sm">
          <h1 className="text-[2rem] leading-tight font-semibold tracking-tight">Teleco-Tutor</h1>
          <p className="mt-3 text-muted text-[1.0625rem] leading-relaxed">
            Ask questions about your course PDFs. Every answer shows the page it came from.
          </p>

          <form onSubmit={submit} className="mt-10 space-y-4" noValidate>
            {DEV_MODE ? (
              <>
                <Field label="Test user" hint="Local development mode: no password. Any name gives you a separate test library.">
                  <input
                    autoFocus
                    value={workspace}
                    onChange={(e) => setWorkspace(e.target.value)}
                    className={inputClass}
                    autoComplete="username"
                  />
                </Field>
              </>
            ) : (
              <>
                <Field label="Email">
                  <input
                    type="email"
                    required
                    autoFocus
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className={inputClass}
                    autoComplete="email"
                  />
                </Field>
                <Field label="Password" hint={mode === "signup" ? "At least 8 characters." : undefined}>
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className={inputClass}
                    autoComplete={mode === "signin" ? "current-password" : "new-password"}
                  />
                </Field>
              </>
            )}

            {error && (
              <p role="alert" className="text-sm text-danger">
                {error}
              </p>
            )}
            {notice && (
              <p role="status" className="text-sm text-ok">
                {notice}
              </p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="w-full rounded-md bg-ink px-4 py-2.5 font-medium text-paper disabled:opacity-60"
            >
              {DEV_MODE ? "Continue" : mode === "signin" ? (busy ? "Signing in…" : "Sign in") : busy ? "Creating account…" : "Create account"}
            </button>
          </form>

          {!DEV_MODE && (
            <p className="mt-6 text-sm text-muted">
              {mode === "signin" ? "New here? " : "Already have an account? "}
              <button
                type="button"
                onClick={() => {
                  setMode(mode === "signin" ? "signup" : "signin");
                  setError(null);
                  setNotice(null);
                }}
                className="font-medium text-ink underline underline-offset-4"
              >
                {mode === "signin" ? "Create an account" : "Sign in"}
              </button>
            </p>
          )}
        </div>
      </section>

      <PagePreview />
    </main>
  );
}

const inputClass =
  "block w-full rounded-md border border-rule bg-surface px-3 py-2.5 text-ink placeholder:text-muted focus:border-ink focus:outline-none";

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-sm text-muted">{hint}</span>}
    </label>
  );
}

/** A drawn course page: the product's core idea - an answer citing a highlighted passage. */
function PagePreview() {
  return (
    <aside aria-hidden className="relative hidden overflow-hidden border-l border-rule bg-surface lg:block">
      <div className="absolute inset-0 [background-image:linear-gradient(var(--rule)_1px,transparent_1px)] [background-size:100%_2.25rem] opacity-60" />
      <div className="relative mx-auto mt-[14vh] w-[min(30rem,80%)]">
        <div className="rounded-sm border border-rule bg-paper px-9 py-8 shadow-[0_1px_0_var(--rule),0_18px_40px_-24px_rgb(0_0_0/0.35)]">
          <p className="text-xs text-muted">Radar range equation, page 22</p>
          <div className="mt-6 space-y-2.5">
            <Line w="92%" />
            <Line w="85%" />
            <p className="font-serif text-[0.95rem] leading-relaxed">
              <span className="bg-mark px-0.5 text-mark-ink box-decoration-clone">
                Three kinds of losses exist in radar systems: losses within the system, losses in the
                propagation path, and losses from ground reflections.
              </span>
            </p>
            <Line w="88%" />
            <Line w="64%" />
          </div>
        </div>
        <div className="answer ml-10 mt-6 rounded-sm border border-rule bg-surface px-6 py-5 text-[0.98rem]">
          The radar equation includes system, propagation and multipath losses
          <span className="cite">1</span>, combined into a single loss factor <i>L</i>.
        </div>
      </div>
    </aside>
  );
}

function Line({ w }: { w: string }) {
  return <div className="h-2 rounded-full bg-rule" style={{ width: w }} />;
}
