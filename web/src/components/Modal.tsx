"use client";

import { useEffect, useRef } from "react";

export function Modal({
  title,
  onClose,
  children,
  wide,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current!;
    if (!dialog.open) dialog.showModal(); // native focus trap, Esc to close
    const close = () => onClose();
    dialog.addEventListener("close", close);
    return () => dialog.removeEventListener("close", close);
  }, [onClose]);

  return (
    <dialog
      ref={ref}
      aria-label={title}
      onClick={(e) => e.target === ref.current && onClose()}
      className={`m-auto w-[calc(100%-2rem)] ${wide ? "max-w-2xl" : "max-w-md"} rounded-lg border border-rule bg-surface p-0 text-ink backdrop:bg-[rgb(10_16_28/0.45)]`}
    >
      <div className="flex items-center justify-between border-b border-rule px-6 py-4">
        <h2 className="text-lg font-semibold">{title}</h2>
        <button onClick={onClose} className="rounded p-1 text-muted hover:text-ink" aria-label="Close">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      <div className="max-h-[75vh] overflow-y-auto px-6 py-5">{children}</div>
    </dialog>
  );
}
