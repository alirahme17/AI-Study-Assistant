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

export type CourseFile = { source: string; name: string };
export type Course = { name: string; files: CourseFile[] };

export type Me = {
  workspace: string;
  status: "not_configured" | "processing" | "ready" | "error";
  message: string;
  has_api_key: boolean;
  embedding_model: string | null;
  courses: Course[];
};

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
    throw new ApiError(0, "Can't reach the Teleco-Tutor server. Check that the backend is running.");
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

  upload: async (h: Headers, course: string, files: File[]): Promise<string[]> => {
    const form = new FormData();
    form.append("course", course);
    files.forEach((f) => form.append("files", f));
    return (await (await request(h, "/api/me/upload", { method: "POST", body: form })).json()).saved;
  },

  deleteFile: (h: Headers, source: string) =>
    request(h, `/api/me/files?source=${encodeURIComponent(source)}`, { method: "DELETE" }),

  index: (h: Headers) => request(h, "/api/me/index", { method: "POST" }),

  importDrive: (h: Headers, url: string) =>
    request(h, "/api/me/drive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drive_folder_url: url }),
    }),

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
  if (/RESOURCE_EXHAUSTED|quota|429/i.test(message))
    return "Your Gemini API key has run out of free quota for today. Try again later, or switch to another model.";
  if (/API key not valid|API_KEY_INVALID|permission/i.test(message))
    return "Gemini rejected your API key. Check it in Settings.";
  if (/not found|404/i.test(message)) return "The Gemini model isn't available for your key. Try another model.";
  return `The answer couldn't be generated: ${message}`;
}
