"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Answer } from "@/components/Answer";
import { LibraryDialog } from "@/components/LibraryDialog";
import { SettingsDialog } from "@/components/SettingsDialog";
import { SourcePanel } from "@/components/SourcePanel";
import { api, type ChatTurn, type Me, type Source } from "@/lib/api";
import { useAuth } from "@/lib/auth";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
  error?: string;
};

const HISTORY_MESSAGES = 6;

export default function Workspace() {
  const { ready, user, authHeaders } = useAuth();
  const router = useRouter();
  const [me, setMe] = useState<Me | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [course, setCourse] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [active, setActive] = useState<{ messageId: string; sourceId: number } | null>(null);
  const [dialog, setDialog] = useState<"library" | "settings" | null>(null);
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    if (ready && !user) router.replace("/login");
  }, [ready, user, router]);

  const refresh = useCallback(
    () =>
      api.me(authHeaders).then(
        (data) => {
          setMe(data);
          setLoadError(null);
        },
        (e: Error) => setLoadError(e.message),
      ),
    [authHeaders],
  );

  useEffect(() => {
    if (!ready || !user) return;
    let cancelled = false;
    api.me(authHeaders).then(
      (data) => !cancelled && (setMe(data), setLoadError(null)),
      (e: Error) => !cancelled && setLoadError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [ready, user, authHeaders]);

  // While the knowledge base is being built, poll for progress.
  useEffect(() => {
    if (me?.status !== "processing") return;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [me?.status, refresh]);

  if (!ready || !user) return null;

  const activeMessage = active && messages.find((m) => m.id === active.messageId);
  const panelOpen = !!(activeMessage && activeMessage.sources?.length);

  return (
    <div
      className={`grid h-dvh min-h-0 grid-cols-1 md:grid-cols-[15.5rem_minmax(0,1fr)] ${
        panelOpen ? "lg:grid-cols-[15.5rem_minmax(0,1fr)_minmax(26rem,44%)]" : ""
      }`}
    >
      <Sidebar
        me={me}
        course={course}
        open={navOpen}
        onClose={() => setNavOpen(false)}
        onCourse={(c) => {
          setCourse(c);
          setNavOpen(false);
        }}
        onLibrary={() => setDialog("library")}
        onSettings={() => setDialog("settings")}
      />

      <main className="flex min-h-0 min-w-0 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-rule px-4 md:px-8">
          <button className="rounded p-1 md:hidden" onClick={() => setNavOpen(true)} aria-label="Open menu">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden>
              <path d="M3 6h16M3 11h16M3 16h16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
          <h1 className="min-w-0 truncate font-medium">{course ?? "All courses"}</h1>
          {messages.length > 0 && (
            <button
              onClick={() => {
                setMessages([]);
                setActive(null);
              }}
              className="ml-auto rounded-md border border-rule px-3 py-1 text-sm text-muted hover:text-ink"
            >
              New chat
            </button>
          )}
        </header>

        {loadError ? (
          <Centered>
            <p role="alert" className="text-danger">{loadError}</p>
            <button onClick={() => refresh()} className="mt-4 rounded-md border border-ink px-4 py-2 text-sm font-medium">
              Try again
            </button>
          </Centered>
        ) : !me ? (
          <Centered>
            <p className="text-muted">Loading your courses…</p>
          </Centered>
        ) : me.status !== "ready" && messages.length === 0 ? (
          <Onboarding me={me} onLibrary={() => setDialog("library")} onSettings={() => setDialog("settings")} />
        ) : (
          <Conversation
            me={me}
            course={course}
            messages={messages}
            setMessages={setMessages}
            active={active}
            onCite={(messageId, sourceId) => setActive({ messageId, sourceId })}
            onSettings={() => setDialog("settings")}
          />
        )}
      </main>

      {panelOpen && (
        <div className="fixed inset-0 z-30 border-l border-rule lg:static lg:z-auto lg:min-h-0">
          <SourcePanel
            sources={activeMessage!.sources!}
            activeId={active!.sourceId}
            onSelect={(sourceId) => setActive({ messageId: active!.messageId, sourceId })}
            onClose={() => setActive(null)}
          />
        </div>
      )}

      {dialog === "library" && me && <LibraryDialog me={me} onClose={() => setDialog(null)} onChanged={refresh} />}
      {dialog === "settings" && me && <SettingsDialog me={me} onClose={() => setDialog(null)} onChanged={refresh} />}
    </div>
  );
}

// ------------------------------------------------------------------ sidebar ---

function Sidebar({
  me,
  course,
  open,
  onClose,
  onCourse,
  onLibrary,
  onSettings,
}: {
  me: Me | null;
  course: string | null;
  open: boolean;
  onClose: () => void;
  onCourse: (c: string | null) => void;
  onLibrary: () => void;
  onSettings: () => void;
}) {
  const courses = me?.courses ?? [];
  return (
    <>
      {open && <div className="fixed inset-0 z-30 bg-[rgb(10_16_28/0.4)] md:hidden" onClick={onClose} aria-hidden />}
      <nav
        aria-label="Courses"
        className={`fixed inset-y-0 left-0 z-40 flex w-[15.5rem] flex-col border-r border-rule bg-paper transition-transform md:static md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="px-5 pt-5 pb-4">
          <p className="text-lg font-semibold tracking-tight">Teleco-Tutor</p>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3">
          <CourseButton label="All courses" count={courses.reduce((n, c) => n + c.files.length, 0)} active={course === null} onClick={() => onCourse(null)} />
          <div className="my-2 border-t border-rule" />
          {courses.map((c) => (
            <CourseButton key={c.name} label={c.name} count={c.files.length} active={course === c.name} onClick={() => onCourse(c.name)} />
          ))}
          {me && courses.length === 0 && <p className="px-2 py-1 text-sm text-muted">No courses yet.</p>}
        </div>

        <div className="space-y-0.5 border-t border-rule px-3 py-3">
          {me?.status === "processing" && (
            <p className="px-2 pb-2 text-xs text-muted" role="status">
              Updating knowledge base…
            </p>
          )}
          <SideAction onClick={onLibrary}>Manage PDFs</SideAction>
          <SideAction onClick={onSettings}>Settings</SideAction>
        </div>
      </nav>
    </>
  );
}

function CourseButton({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-current={active}
      className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-[0.94rem] ${
        active ? "bg-surface font-medium shadow-[inset_3px_0_0_var(--mark)]" : "text-muted hover:text-ink"
      }`}
    >
      <span className="min-w-0 truncate">{label}</span>
      <span className="ml-2 shrink-0 text-xs tabular-nums text-muted">{count}</span>
    </button>
  );
}

function SideAction({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} className="w-full rounded-md px-2 py-1.5 text-left text-[0.94rem] text-muted hover:bg-surface hover:text-ink">
      {children}
    </button>
  );
}

// --------------------------------------------------------------- onboarding ---

function Onboarding({ me, onLibrary, onSettings }: { me: Me; onLibrary: () => void; onSettings: () => void }) {
  const hasFiles = me.courses.some((c) => c.files.length > 0);
  const steps = [
    { done: hasFiles, title: "Add your course PDFs", text: "Slides, lecture notes, past exams. Group them by course.", action: "Add PDFs", onClick: onLibrary },
    {
      done: me.status === "ready",
      title: "Build your knowledge base",
      text: me.status === "processing" ? me.message || "Reading your PDFs…" : "The tutor reads every page once, including scanned ones.",
      action: me.status === "processing" ? null : "Build it",
      onClick: onLibrary,
    },
    { done: me.has_api_key, title: "Add your Gemini API key", text: "Answers are written by Google Gemini with your own free key.", action: "Add key", onClick: onSettings },
  ];
  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-10 md:px-8">
      <div className="mx-auto max-w-xl">
        <h2 className="text-2xl font-semibold tracking-tight">Set up your tutor</h2>
        <p className="mt-2 text-muted">Three steps, then ask anything about your courses.</p>
        {me.status === "error" && (
          <p role="alert" className="mt-4 rounded-md border border-danger/40 px-4 py-3 text-sm text-danger">
            The last update failed: {me.message}
          </p>
        )}
        <ol className="mt-8 space-y-3">
          {steps.map((s, i) => (
            <li key={s.title} className="flex gap-4 rounded-lg border border-rule bg-surface px-5 py-4">
              <span
                className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-semibold ${
                  s.done ? "bg-mark text-mark-ink" : "border border-rule text-muted"
                }`}
                aria-label={s.done ? "Done" : `Step ${i + 1}`}
              >
                {s.done ? "✓" : i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-medium">{s.title}</p>
                <p className="mt-0.5 text-sm text-muted" role={me.status === "processing" && i === 1 ? "status" : undefined}>
                  {s.text}
                </p>
              </div>
              {s.action && !s.done && (
                <button onClick={s.onClick} className="self-center rounded-md border border-ink px-3 py-1.5 text-sm font-medium">
                  {s.action}
                </button>
              )}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- conversation ---

function Conversation({
  me,
  course,
  messages,
  setMessages,
  active,
  onCite,
  onSettings,
}: {
  me: Me;
  course: string | null;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  active: { messageId: string; sourceId: number } | null;
  onCite: (messageId: string, sourceId: number) => void;
  onSettings: () => void;
}) {
  const { authHeaders } = useAuth();
  const [draft, setDraft] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const streaming = messages.some((m) => m.streaming);

  // Follow the answer as it streams, unless the reader scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const update = (id: string, patch: (m: Message) => Partial<Message>) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));

  async function send() {
    const question = draft.trim();
    if (!question || streaming) return;
    const history: ChatTurn[] = messages
      .filter((m) => !m.error && m.content)
      .slice(-HISTORY_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content }));
    const answerId = crypto.randomUUID();
    setDraft("");
    setMessages((ms) => [
      ...ms,
      { id: crypto.randomUUID(), role: "user", content: question },
      { id: answerId, role: "assistant", content: "", streaming: true },
    ]);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await api.chat(authHeaders, { question, course, history }, {
        signal: controller.signal,
        onSources: (sources) => update(answerId, () => ({ sources })),
        onToken: (t) => update(answerId, (m) => ({ content: m.content + t })),
      });
    } catch (e) {
      if ((e as Error).name !== "AbortError") update(answerId, () => ({ error: (e as Error).message }));
    } finally {
      update(answerId, () => ({ streaming: false }));
      abortRef.current = null;
    }
  }

  return (
    <>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 md:px-8">
        {messages.length === 0 ? (
          <div className="mx-auto flex h-full max-w-2xl flex-col justify-center py-10">
            <h2 className="text-2xl font-semibold tracking-tight">
              {course ? `Ask about ${course}` : "Ask about your courses"}
            </h2>
            <p className="mt-2 max-w-prose text-muted">
              Answers come only from your PDFs. Click a highlighted number to see the page it came from.
            </p>
            {!me.has_api_key && (
              <p className="mt-6 text-sm">
                Add your Gemini API key in{" "}
                <button onClick={onSettings} className="underline underline-offset-2">Settings</button> to get answers.
              </p>
            )}
          </div>
        ) : (
          <ol className="mx-auto max-w-3xl space-y-8 py-8">
            {messages.map((m) =>
              m.role === "user" ? (
                <li key={m.id} className="flex justify-end">
                  <p className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-surface px-4 py-2.5 shadow-[0_0_0_1px_var(--rule)]">
                    {m.content}
                  </p>
                </li>
              ) : (
                <li key={m.id} aria-live={m.streaming ? "polite" : undefined}>
                  {m.error ? (
                    <p role="alert" className="text-danger">{m.error}</p>
                  ) : m.content ? (
                    <Answer
                      text={m.content}
                      streaming={m.streaming}
                      validIds={new Set((m.sources ?? []).map((s) => s.id))}
                      activeId={active?.messageId === m.id ? active.sourceId : null}
                      onCite={(id) => onCite(m.id, id)}
                    />
                  ) : (
                    <p className="text-muted">{m.sources ? "Writing the answer…" : "Searching your PDFs…"}</p>
                  )}
                  {!m.streaming && !m.error && m.sources && m.sources.length > 0 && (
                    <SourceList message={m} active={active} onCite={onCite} />
                  )}
                </li>
              ),
            )}
          </ol>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="border-t border-rule px-4 py-3 md:px-8"
      >
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-lg border border-rule bg-surface px-3 py-2 focus-within:border-ink">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={Math.min(6, Math.max(1, draft.split("\n").length))}
            placeholder={course ? `Ask a question about ${course}` : "Ask a question about your courses"}
            aria-label="Your question"
            className="max-h-48 min-h-[2.25rem] flex-1 resize-none bg-transparent py-1.5 focus:outline-none"
          />
          {streaming ? (
            <button type="button" onClick={() => abortRef.current?.abort()} className="rounded-md border border-rule px-3 py-1.5 text-sm">
              Stop
            </button>
          ) : (
            <button disabled={!draft.trim()} className="rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-paper disabled:opacity-40">
              Ask
            </button>
          )}
        </div>
        <p className="mx-auto mt-1.5 max-w-3xl text-xs text-muted">
          Searching {course ?? "all courses"}. Shift+Enter for a new line.
        </p>
      </form>
    </>
  );
}

/** The sources an answer cited (or, if it cited none, everything that was retrieved). */
function SourceList({
  message,
  active,
  onCite,
}: {
  message: Message;
  active: { messageId: string; sourceId: number } | null;
  onCite: (messageId: string, sourceId: number) => void;
}) {
  const cited = new Set(
    Array.from(message.content.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)).flatMap((m) => m[1].split(",").map((n) => Number(n.trim()))),
  );
  const shown = message.sources!.filter((s) => cited.has(s.id));
  const list = shown.length ? shown : message.sources!;
  return (
    <div className="mt-4 border-t border-rule pt-3">
      <p className="text-xs text-muted">{shown.length ? "Sources" : "Searched these pages, none cited"}</p>
      <ul className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
        {list.map((s) => (
          <li key={s.id}>
            <button
              onClick={() => onCite(message.id, s.id)}
              className={`text-left text-sm hover:text-ink ${
                active?.messageId === message.id && active.sourceId === s.id ? "text-ink" : "text-muted"
              }`}
            >
              <span className="cite mr-1">{s.id}</span>
              {s.file_name}
              {s.page ? `, p. ${s.page}` : ""}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">{children}</div>;
}

