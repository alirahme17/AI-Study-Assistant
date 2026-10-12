"use client";

import { useState } from "react";
import { api, type CourseFile, type Me, type UploadItem } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Modal } from "./Modal";

const isPdf = (f: File) => f.name.toLowerCase().endsWith(".pdf");

/** Where each picked file goes. A picked folder keeps its sub-folders; with no course
 * typed, each sub-folder of the picked folder becomes its own course. */
function plan(files: File[], course: string): UploadItem[] {
  return files.filter(isPdf).map((file) => {
    const parts = (file.webkitRelativePath || file.name).split("/").filter(Boolean);
    if (parts.length > 1) parts.shift(); // drop the picked folder's own name
    if (course) return { file, course, name: parts.join("/") };
    if (parts.length > 1) return { file, course: parts[0], name: parts.slice(1).join("/") };
    const root = file.webkitRelativePath.split("/")[0];
    return { file, course: root || "General", name: parts[0] };
  });
}

function summarize(items: UploadItem[]) {
  const by = new Map<string, number>();
  items.forEach((i) => by.set(i.course, (by.get(i.course) ?? 0) + 1));
  return Array.from(by, ([c, n]) => `${c} (${n})`).join(", ");
}

/** Add, remove and re-index the course PDFs. Uploads are indexed automatically. */
export function LibraryDialog({ me, onClose, onChanged }: { me: Me; onClose: () => void; onChanged: () => void }) {
  const { authHeaders } = useAuth();
  const [course, setCourse] = useState(me.courses[0]?.name ?? "");
  const [picked, setPicked] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const items = plan(picked, course.trim());
  const tooBig = items.filter((i) => i.file.size > me.max_upload_mb * 1024 * 1024);
  const files = me.courses.flatMap((c) => c.files);
  const failed = files.filter((f) => f.status === "error");
  const paused = files.some((f) => f.status === "pending" && f.message.startsWith("Paused"));

  async function run(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
      setProgress(null);
      onChanged();
    }
  }

  function pick(list: FileList | null) {
    const all = Array.from(list ?? []);
    const pdfs = all.filter(isPdf);
    setPicked(pdfs);
    setError(all.length > pdfs.length ? `${all.length - pdfs.length} non-PDF file(s) skipped.` : null);
  }

  async function upload() {
    const todo = items.filter((i) => !tooBig.includes(i));
    const failures: string[] = [];
    for (const [n, item] of todo.entries()) {
      setProgress(`Uploading ${n + 1} of ${todo.length}: ${item.name}`);
      try {
        await api.upload(authHeaders, item);
      } catch (e) {
        failures.push(`${item.name}: ${(e as Error).message}`);
      }
      if (n % 5 === 4) onChanged(); // show indexing progress while the rest uploads
    }
    setPicked([]);
    if (failures.length) throw new Error(`${failures.length} file(s) failed. ${failures.slice(0, 3).join(" · ")}`);
  }

  return (
    <Modal title="Your courses" onClose={onClose} wide>
      <section>
        <h3 className="font-medium">Add PDFs</h3>
        <p className="mt-1 text-sm text-muted">They&apos;re read and indexed automatically after upload.</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-[12rem_1fr]">
          <label className="block">
            <span className="mb-1 block text-sm text-muted">Course</span>
            <input
              list="course-names"
              value={course}
              onChange={(e) => setCourse(e.target.value)}
              placeholder="e.g. Radar"
              className="w-full rounded-md border border-rule bg-paper px-3 py-2 focus:border-ink focus:outline-none"
            />
            <datalist id="course-names">
              {me.courses.map((c) => (
                <option key={c.name} value={c.name} />
              ))}
            </datalist>
          </label>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pick(e.dataTransfer.files);
            }}
            className={`flex flex-col justify-center gap-2 rounded-md border border-dashed px-4 py-3 text-sm ${
              dragging ? "border-ink bg-paper" : "border-rule"
            }`}
          >
            {picked.length ? (
              <span>
                {items.length} PDF{items.length > 1 ? "s" : ""} → <span className="text-muted">{summarize(items)}</span>
              </span>
            ) : (
              <span className="text-muted">Drop PDFs here, or choose:</span>
            )}
            <div className="flex flex-wrap gap-2">
              <label className="cursor-pointer rounded border border-rule px-2.5 py-1 hover:border-ink">
                <input type="file" accept="application/pdf,.pdf" multiple className="sr-only" onChange={(e) => pick(e.target.files)} />
                Files…
              </label>
              <label className="cursor-pointer rounded border border-rule px-2.5 py-1 hover:border-ink">
                <input
                  type="file"
                  multiple
                  className="sr-only"
                  onChange={(e) => pick(e.target.files)}
                  {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
                />
                A whole folder…
              </label>
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs text-muted">
          Folder tip: leave the course empty to turn each sub-folder into its own course.
        </p>
        {tooBig.length > 0 && (
          <p className="mt-2 text-sm text-danger">
            Skipping {tooBig.length} file(s) over {me.max_upload_mb} MB: {tooBig.map((i) => i.file.name).join(", ")}
          </p>
        )}
        <button
          disabled={!items.length || !!busy}
          onClick={() => run("upload", upload)}
          className="mt-3 rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper disabled:opacity-50"
        >
          {busy === "upload" ? "Uploading…" : items.length ? `Upload ${items.length} PDF${items.length > 1 ? "s" : ""}` : "Upload"}
        </button>
        {progress && (
          <p role="status" className="mt-2 truncate text-sm text-muted">
            {progress}
          </p>
        )}
      </section>

      {error && (
        <p role="alert" className="mt-4 text-sm text-danger">
          {error}
        </p>
      )}

      {(me.status === "processing" || paused) && (
        <p role="status" className="mt-6 rounded-md bg-paper px-4 py-3 text-sm">
          {me.message}
        </p>
      )}

      <section className="mt-6">
        <div className="flex items-center justify-between gap-3">
          <h3 className="font-medium">
            Files <span className="font-normal text-muted">({files.length})</span>
          </h3>
          {(failed.length > 0 || paused) && (
            <button
              disabled={!!busy}
              onClick={() => run("reindex", () => api.reindex(authHeaders))}
              className="rounded-md border border-ink px-3 py-1 text-sm font-medium disabled:opacity-50"
            >
              {paused ? "Resume indexing" : `Retry ${failed.length} failed`}
            </button>
          )}
        </div>
        {me.courses.length === 0 ? (
          <p className="mt-2 text-sm text-muted">No PDFs yet. Add a course&apos;s slides, notes or past exams above.</p>
        ) : (
          <div className="mt-2 divide-y divide-rule border-y border-rule">
            {me.courses.map((c) => {
              const ready = c.files.filter((f) => f.status === "ready").length;
              return (
                <details key={c.name} className="group py-2" open={me.courses.length === 1}>
                  <summary className="flex cursor-pointer list-none items-center justify-between py-1">
                    <span>{c.name}</span>
                    <span className="text-sm text-muted tabular-nums">
                      {ready === c.files.length ? `${c.files.length} files` : `${ready}/${c.files.length} indexed`}
                    </span>
                  </summary>
                  <ul className="mt-1 mb-1 space-y-0.5">
                    {c.files.map((f) => (
                      <FileRow
                        key={f.source}
                        file={f}
                        disabled={!!busy}
                        onRemove={() => run("delete", () => api.deleteFile(authHeaders, f.source))}
                        onRetry={() => run("reindex", () => api.reindex(authHeaders, f.source))}
                      />
                    ))}
                  </ul>
                </details>
              );
            })}
          </div>
        )}
      </section>
    </Modal>
  );
}

const STATUS: Record<CourseFile["status"], { label: string; className: string }> = {
  ready: { label: "Indexed", className: "text-ok" },
  processing: { label: "Indexing…", className: "text-ink" },
  pending: { label: "Waiting", className: "text-muted" },
  error: { label: "Failed", className: "text-danger" },
};

function FileRow({
  file,
  disabled,
  onRemove,
  onRetry,
}: {
  file: CourseFile;
  disabled: boolean;
  onRemove: () => void;
  onRetry: () => void;
}) {
  const status = file.message.startsWith("Paused") ? { label: "Paused", className: "text-muted" } : STATUS[file.status];
  const detail = file.status === "processing" || file.status === "error" || file.message.startsWith("Paused");
  return (
    <li className="py-1 pl-3 text-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="min-w-0 truncate" title={file.source}>
          {file.name.split("/").join(" / ")}
        </span>
        <span className="flex shrink-0 items-center gap-3">
          <span className={`text-xs ${status.className}`} title={file.message}>
            {status.label}
            {file.status === "ready" && file.pages ? <span className="text-muted"> · {file.pages} p.</span> : null}
          </span>
          {file.status === "error" && (
            <button disabled={disabled} onClick={onRetry} className="text-muted hover:text-ink disabled:opacity-40">
              Retry
            </button>
          )}
          <button
            disabled={disabled || file.status === "processing"}
            onClick={onRemove}
            className="text-muted hover:text-danger disabled:opacity-40"
          >
            Remove
          </button>
        </span>
      </div>
      {detail && file.message && <p className="mt-0.5 truncate text-xs text-muted">{file.message}</p>}
    </li>
  );
}
