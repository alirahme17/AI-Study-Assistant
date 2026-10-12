"use client";

import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

/**
 * Render an answer (Markdown + LaTeX) and turn citations like [2] or [1, 3]
 * into highlighter chips that open the cited page.
 */
export function Answer({
  text,
  streaming,
  activeId,
  validIds,
  onCite,
}: {
  text: string;
  streaming?: boolean;
  activeId?: number | null;
  validIds: Set<number>;
  onCite: (id: number) => void;
}) {
  // [1, 3] -> [1](#cite-1)[3](#cite-3); only numbers that match a real source become chips.
  const withLinks = text.replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (whole, group: string) => {
    const ids = group.split(",").map((n) => Number(n.trim()));
    if (!ids.every((id) => validIds.has(id))) return whole;
    return ids.map((id) => `[${id}](#cite-${id})`).join("");
  });

  return (
    <div className={`answer ${streaming ? "caret" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a({ href, children }) {
            const match = href?.match(/^#cite-(\d+)$/);
            if (!match) {
              return (
                <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
                  {children}
                </a>
              );
            }
            const id = Number(match[1]);
            return (
              <button
                type="button"
                className="cite"
                aria-pressed={activeId === id}
                aria-label={`Show source ${id}`}
                onClick={() => onCite(id)}
              >
                {id}
              </button>
            );
          },
        }}
      >
        {withLinks}
      </ReactMarkdown>
    </div>
  );
}
