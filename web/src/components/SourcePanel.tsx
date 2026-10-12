"use client";

import { useEffect, useRef, useState } from "react";
import { api, type Source } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/** The cited PDF page, with the passage the answer used highlighted. */
export function SourcePanel({
  sources,
  activeId,
  onSelect,
  onClose,
}: {
  sources: Source[];
  activeId: number;
  onSelect: (id: number) => void;
  onClose: () => void;
}) {
  const { authHeaders } = useAuth();
  const source = sources.find((s) => s.id === activeId) ?? sources[0];
  const sourceKey = source ? `${source.source}#${source.page}` : "";

  // Page being viewed, remembered per source: selecting another source jumps to its cited page.
  const [viewed, setViewed] = useState<{ key: string; page: number } | null>(null);
  const page = viewed?.key === sourceKey ? viewed.page : source?.page ?? 1;
  const setPage = (update: (p: number) => number) => setViewed({ key: sourceKey, page: update(page) });

  // Result of the last image request; it belongs to the current view only if its key matches.
  const requestKey = `${sourceKey}@${page}`;
  const [result, setResult] = useState<{ key: string; url?: string; pageCount?: number | null; error?: string } | null>(null);
  const [lastImage, setLastImage] = useState<{ url: string; pageCount: number | null } | null>(null);
  const loading = result?.key !== requestKey;
  const error = !loading ? result?.error ?? null : null;
  const image = !loading && result?.url ? { url: result.url, pageCount: result.pageCount ?? null } : lastImage;

  // Object URLs are released when replaced by a newer image, or when the panel closes;
  // the previous image stays visible (dimmed) while the next one loads.
  const shownUrl = useRef<string | null>(null);
  useEffect(() => () => {
    if (shownUrl.current) URL.revokeObjectURL(shownUrl.current);
  }, []);

  useEffect(() => {
    if (!source) return;
    let cancelled = false;
    const highlight = page === source.page ? source.snippet : undefined;
    api.pageImage(authHeaders, source.source, page, highlight).then(
      (img) => {
        if (cancelled) {
          URL.revokeObjectURL(img.url);
          return;
        }
        if (shownUrl.current) URL.revokeObjectURL(shownUrl.current);
        shownUrl.current = img.url;
        setResult({ key: requestKey, url: img.url, pageCount: img.pageCount });
        setLastImage(img);
      },
      (e: Error) => !cancelled && setResult({ key: requestKey, error: e.message }),
    );
    return () => {
      cancelled = true;
    };
  }, [authHeaders, source, page, requestKey]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!source) return null;
  const folder = source.source.split("/").slice(0, -1).join(" / ");

  return (
    <section aria-label="Source" className="flex h-full min-h-0 flex-col bg-surface">
      <header className="border-b border-rule px-5 pt-4 pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="truncate font-medium" title={source.file_name}>
              {source.file_name}
            </h2>
            <p className="truncate text-sm text-muted" title={folder}>
              {folder}
            </p>
          </div>
          <button onClick={onClose} className="-mr-1 rounded p-1 text-muted hover:text-ink" aria-label="Close source">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
              <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {sources.length > 1 && (
          <nav aria-label="Sources of this answer" className="mt-3 flex flex-wrap gap-1.5">
            {sources.map((s) => (
              <button
                key={s.id}
                onClick={() => onSelect(s.id)}
                aria-pressed={s.id === source.id}
                className={`rounded px-2 py-1 text-xs tabular-nums transition-colors ${
                  s.id === source.id ? "bg-mark text-mark-ink" : "bg-paper text-muted hover:text-ink"
                }`}
                title={`${s.file_name}${s.page ? `, page ${s.page}` : ""}`}
              >
                [{s.id}] p.{s.page ?? "?"}
              </button>
            ))}
          </nav>
        )}
      </header>

      <div className="flex items-center justify-between border-b border-rule px-5 py-2 text-sm">
        <button
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1}
          className="rounded px-2 py-1 text-muted hover:text-ink disabled:opacity-40"
        >
          Previous page
        </button>
        <span className="tabular-nums text-muted">
          Page {page}
          {image?.pageCount ? ` of ${image.pageCount}` : ""}
          {page !== source.page && source.page && (
            <>
              {" "}
              <button onClick={() => setPage(() => source.page!)} className="ml-1 text-ink underline underline-offset-2">
                back to cited page
              </button>
            </>
          )}
        </span>
        <button
          onClick={() => setPage((p) => p + 1)}
          disabled={image?.pageCount != null && page >= image.pageCount}
          className="rounded px-2 py-1 text-muted hover:text-ink disabled:opacity-40"
        >
          Next page
        </button>
      </div>

      {source.ocr && page === source.page && (
        <p className="border-b border-rule bg-paper px-5 py-2 text-sm text-muted">
          This page is a scan. Its text was read with OCR, so the passage can&apos;t be highlighted and may contain
          recognition errors.
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-auto bg-paper p-4">
        {error ? (
          <p role="alert" className="text-sm text-danger">
            {error}
          </p>
        ) : image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={image.url}
            alt={`Page ${page} of ${source.file_name}`}
            className={`mx-auto w-full max-w-[760px] border border-rule bg-white shadow-sm transition-opacity ${
              loading ? "opacity-50" : ""
            }`}
          />
        ) : (
          <div className="mx-auto aspect-[1/1.41] w-full max-w-[760px] animate-pulse border border-rule bg-surface" />
        )}
      </div>
    </section>
  );
}
