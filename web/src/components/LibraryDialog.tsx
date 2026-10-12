"use client";

import { useState } from "react";
import { api, type Me } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Modal } from "./Modal";

/** Add, remove and index the course PDFs. */
export function LibraryDialog({ me, onClose, onChanged }: { me: Me; onClose: () => void; onChanged: () => void }) {
  const { authHeaders } = useAuth();
  const [course, setCourse] = useState(me.courses[0]?.name ?? "");
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [changed, setChanged] = useState(false);
  const [driveUrl, setDriveUrl] = useState("");
  const processing = me.status === "processing";

  async function run(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  function pick(list: FileList | null) {
    const pdfs = Array.from(list ?? []).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    setFiles(pdfs);
    if (list && pdfs.length < list.length) setError("Only PDF files can be added; the others were skipped.");
  }

  const fileCount = me.courses.reduce((n, c) => n + c.files.length, 0);

  return (
    <Modal title="Your courses" onClose={onClose} wide>
      <section>
        <h3 className="font-medium">Add PDFs</h3>
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
          <label
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
            className={`flex cursor-pointer flex-col justify-center rounded-md border border-dashed px-4 py-3 text-sm ${
              dragging ? "border-ink bg-paper" : "border-rule"
            }`}
          >
            <input type="file" accept="application/pdf,.pdf" multiple className="sr-only" onChange={(e) => pick(e.target.files)} />
            {files.length ? (
              <span>
                {files.length} PDF{files.length > 1 ? "s" : ""} selected:{" "}
                <span className="text-muted">{files.map((f) => f.name).join(", ")}</span>
              </span>
            ) : (
              <span className="text-muted">Drop PDFs here, or click to choose files</span>
            )}
          </label>
        </div>
        <button
          disabled={!files.length || !course.trim() || !!busy}
          onClick={() =>
            run("upload", async () => {
              await api.upload(authHeaders, course.trim(), files);
              setFiles([]);
              setChanged(true);
            })
          }
          className="mt-3 rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper disabled:opacity-50"
        >
          {busy === "upload" ? "Uploading…" : `Add to ${course.trim() || "course"}`}
        </button>
      </section>

      {error && (
        <p role="alert" className="mt-4 text-sm text-danger">
          {error}
        </p>
      )}

      <section className="mt-8">
        <h3 className="font-medium">
          Files <span className="font-normal text-muted">({fileCount})</span>
        </h3>
        {me.courses.length === 0 ? (
          <p className="mt-2 text-sm text-muted">No PDFs yet. Add a course&apos;s slides, notes or past exams above.</p>
        ) : (
          <div className="mt-2 divide-y divide-rule border-y border-rule">
            {me.courses.map((c) => (
              <details key={c.name} className="group py-2">
                <summary className="flex cursor-pointer list-none items-center justify-between py-1">
                  <span>{c.name}</span>
                  <span className="text-sm text-muted tabular-nums">{c.files.length} files</span>
                </summary>
                <ul className="mt-1 mb-1 space-y-0.5">
                  {c.files.map((f) => (
                    <li key={f.source} className="flex items-center justify-between gap-3 py-1 pl-3 text-sm">
                      <span className="min-w-0 truncate text-muted" title={f.source}>
                        {f.source.split("/").slice(1).join(" / ")}
                      </span>
                      <button
                        disabled={!!busy || processing}
                        onClick={() =>
                          run("delete", async () => {
                            await api.deleteFile(authHeaders, f.source);
                            setChanged(true);
                          })
                        }
                        className="shrink-0 text-muted hover:text-danger disabled:opacity-40"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              </details>
            ))}
          </div>
        )}
      </section>

      <section className="mt-8 rounded-md bg-paper px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm">
            {processing ? (
              <span>{me.message || "Updating your knowledge base…"}</span>
            ) : changed || me.status !== "ready" ? (
              <span>Update the knowledge base so the tutor can use your latest files.</span>
            ) : (
              <span className="text-muted">The knowledge base is up to date.</span>
            )}
          </p>
          <button
            disabled={processing || !!busy || fileCount === 0}
            onClick={() => run("index", async () => {
              await api.index(authHeaders);
              setChanged(false);
            })}
            className="rounded-md border border-ink px-3 py-1.5 text-sm font-medium disabled:opacity-50"
          >
            {processing ? "Updating…" : "Update knowledge base"}
          </button>
        </div>
        {me.status === "error" && !processing && (
          <p className="mt-2 text-sm text-danger">The last update failed: {me.message}</p>
        )}
      </section>

      <details className="mt-6 text-sm">
        <summary className="cursor-pointer text-muted hover:text-ink">Import a Google Drive folder instead</summary>
        <p className="mt-2 text-muted">
          Works when the server runs on your own computer: Google asks you to sign in there.
        </p>
        <div className="mt-2 flex gap-2">
          <input
            value={driveUrl}
            onChange={(e) => setDriveUrl(e.target.value)}
            placeholder="https://drive.google.com/drive/folders/…"
            className="min-w-0 flex-1 rounded-md border border-rule bg-paper px-3 py-2 focus:border-ink focus:outline-none"
          />
          <button
            disabled={!driveUrl.trim() || !!busy || processing}
            onClick={() => run("drive", () => api.importDrive(authHeaders, driveUrl.trim()))}
            className="rounded-md border border-ink px-3 py-2 font-medium disabled:opacity-50"
          >
            Import
          </button>
        </div>
      </details>
    </Modal>
  );
}
