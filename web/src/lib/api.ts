import { API_URL } from "./config";

export type Source = {
  id: number;
  source: string;      // "Radar/Course/Chapter 1/chap1.pdf"
  file_name: string;
  course: string | null;
  page: number | null;
  ocr: boolean;
  snippet: string;
};

export type FileStatus = "pending" | "processing" | "ready" | "error";

export type CourseFile = {
  source: string;      // "Radar/Course/chap1.pdf"
  course: string;
  name: string;        // path inside the course: "Course/chap1.pdf"
  file_name: string;   // "chap1.pdf"
  status: FileStatus;
  message: string;
  pages: number | null;
  chunks: number;
  ocr_pages: number;
  size_bytes: number | null;
};
export type Course = { name: string; files: CourseFile[] };

export type Me = {
  user_id: string;
  status: "empty" | "processing" | "ready" | "error";
  message: string;
  indexing: boolean;
  has_api_key: boolean;
  courses: Course[];
  max_upload_mb: number;
};

/** A file to upload, with its path inside the course ("Chapter 1/slides.pdf"). */
export type UploadItem = { file: File; course: string; name: string };

export type ChatTurn = { role: "user" | "assistant"; content: string };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

type Headers = () => Promise<Record<string, string>>;

async function request(headers: Headers, path: string, init: RequestInit = {}): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { ...(await headers()), ...(init.headers as Record<string, string> | undefined) },
    });
  } catch {
    throw new ApiError(
      0,
      "Can't reach the Teleco-Tutor server. If it was asleep (free hosting sleeps when idle), it can take up to a minute to wake up. Try again.",
    );
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {}
    throw new ApiError(res.status, detail);
  }
  return res;
}

export const api = {
  me: async (h: Headers): Promise<Me> => (await request(h, "/api/me")).json(),

  saveKey: (h: Headers, key: string) =>
    request(h, "/api/me/key", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gemini_api_key: key }),
    }),

  /** Upload one PDF into a course; indexing starts by itself on the server. */
  upload: async (h: Headers, item: UploadItem): Promise<string[]> => {
    const form = new FormData();
    form.append("course", item.course);
    form.append("files", item.file, item.name);
    return (await (await request(h, "/api/me/upload", { method: "POST", body: form })).json()).saved;
  },

  deleteFile: (h: Headers, source: string) =>
    request(h, `/api/me/files?source=${encodeURIComponent(source)}`, { method: "DELETE" }),

  /** Index one file again (or every file), e.g. after a failure or the daily limit. */
  reindex: (h: Headers, source?: string) =>
    request(h, `/api/me/index${source ? `?source=${encodeURIComponent(source)}` : ""}`, { method: "POST" }),

  /** Object URL of a rendered page (with the cited passage highlighted) + page count. */
  pageImage: async (h: Headers, source: string, page: number, highlight?: string) => {
    const params = new URLSearchParams({ source, page: String(page) });
    if (highlight) params.set("highlight", highlight.slice(0, 400));
    const res = await request(h, `/api/me/page?${params}`);
    return {
      url: URL.createObjectURL(await res.blob()),
      pageCount: Number(res.headers.get("X-Page-Count")) || null,
    };
  },

  /** Ask a question; calls onSources once, then onToken for every streamed piece. */
  chat: async (
    h: Headers,
    body: { question: string; course: string | null; history: ChatTurn[] },
    handlers: { onSources: (s: Source[]) => void; onToken: (t: string) => void; signal?: AbortSignal },
  ) => {
    const res = await request(h, "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: handlers.signal,
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === "sources") handlers.onSources(event.sources);
        else if (event.type === "token") handlers.onToken(event.text);
        else if (event.type === "error") throw new ApiError(502, friendlyModelError(event.message));
      }
    }
  },
};

/** Turn raw Gemini errors into something a student can act on. */
export function friendlyModelError(message: string): string {
  if (/RESOURCE_EXHAUSTED|quota|429|daily limit/i.test(message))
    return "Your Gemini API key has run out of free quota for today. Try again later, or switch to another model.";
  if (/API key not valid|API_KEY_INVALID|permission/i.test(message))
    return "Gemini rejected your API key. Check it in Settings.";
  if (/not found|404/i.test(message)) return "The Gemini model isn't available for your key. Try another model.";
  return `The answer couldn't be generated: ${message}`;
}
