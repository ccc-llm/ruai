// A small Markdown renderer for notebook cells: headings, paragraphs,
// emphasis, inline and fenced code, lists, block quotes, links, rules.
import { esc } from "./ui.mjs";

function inline(s) {
  const codes = [];
  s = esc(s).replace(/`([^`]+)`/g, (_, c) => `\u0000${codes.push(c) - 1}\u0000`);
  s = s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => (/^(https?:|#|\.?\/)/.test(u) ? `<a href="${u}" target="_blank" rel="noopener">${t}</a>` : t));
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

/** HTML for Markdown text. */
export function markdown(src) {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  let out = "", para = [], list = null;
  const endPara = () => {
    if (para.length) out += `<p>${inline(para.join(" "))}</p>`;
    para = [];
  };
  const endList = () => {
    if (list) out += `</${list}>`;
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (/^```/.test(l)) {
      endPara();
      endList();
      const code = [];
      while (++i < lines.length && !/^```/.test(lines[i])) code.push(lines[i]);
      out += `<pre><code>${esc(code.join("\n"))}</code></pre>`;
    } else if ((m = /^(#{1,6})\s+(.*)$/.exec(l))) {
      endPara();
      endList();
      out += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`;
    } else if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) {
      endPara();
      endList();
      out += "<hr>";
    } else if ((m = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(l))) {
      endPara();
      const kind = /\d/.test(m[1]) ? "ol" : "ul";
      if (list !== kind) {
        endList();
        out += `<${kind}>`;
        list = kind;
      }
      out += `<li>${inline(m[2])}</li>`;
    } else if ((m = /^>\s?(.*)$/.exec(l))) {
      endPara();
      endList();
      out += `<blockquote>${inline(m[1])}</blockquote>`;
    } else if (!l.trim()) {
      endPara();
      endList();
    } else {
      endList();
      para.push(l.trim());
    }
  }
  endPara();
  endList();
  return out;
}
