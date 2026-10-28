// Render the ```mermaid code blocks of the manual (GitHub renders them natively). Mermaid is
// imported as an ES module, so that it does not interfere with the RequireJS loader of Documenter.
document.addEventListener("DOMContentLoaded", async () => {
  const blocks = document.querySelectorAll("pre > code.language-mermaid");
  if (blocks.length === 0) {
    return;
  }
  const { default: mermaid } = await import(
    "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"
  );
  for (const code of blocks) {
    const div = document.createElement("div");
    div.className = "mermaid";
    div.textContent = code.textContent;
    code.parentElement.replaceWith(div);
  }
  const dark = document.documentElement.classList.contains("theme--documenter-dark");
  mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "default" });
  await mermaid.run({ querySelector: "div.mermaid" });
});
