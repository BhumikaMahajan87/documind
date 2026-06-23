const $ = (id) => document.getElementById(id);

async function refreshStats() {
  try {
    const res = await fetch("/api/stats");
    const s = await res.json();
    $("provider").textContent =
      `Provider: ${s.provider} | embedder: ${s.embedder} | ${s.documents} docs / ${s.chunks} chunks`;
  } catch (_) {}
}

async function refreshDocs() {
  try {
    const res = await fetch("/api/documents");
    const docs = await res.json();
    $("docList").innerHTML = docs
      .map((d) => `<li>[doc] ${d.doc_name} <span class="badge">${d.chunks} chunks</span></li>`)
      .join("");
  } catch (_) {}
}

$("uploadBtn").addEventListener("click", async () => {
  const fileInput = $("file");
  if (!fileInput.files.length) {
    $("uploadStatus").textContent = "Please choose a file first.";
    return;
  }
  const btn = $("uploadBtn");
  btn.disabled = true;
  $("uploadStatus").textContent = "Uploading & indexing...";
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  try {
    const res = await fetch("/api/documents", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    $("uploadStatus").textContent = `Indexed "${data.doc_name}" into ${data.chunks} chunks.`;
    await refreshDocs();
    await refreshStats();
  } catch (e) {
    $("uploadStatus").textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
});

$("askBtn").addEventListener("click", ask);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

async function ask() {
  const question = $("question").value.trim();
  if (!question) return;
  const btn = $("askBtn");
  btn.disabled = true;
  $("result").style.display = "block";
  $("answer").textContent = "Thinking...";
  $("citations").innerHTML = "";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    $("answer").textContent = data.answer;
    $("citations").innerHTML = data.citations
      .map(
        (c) =>
          `<div class="citation"><span class="src">[${c.ref}] ${c.doc_name}</span>
           <span class="badge">score ${c.score}</span><br/>${c.preview}</div>`
      )
      .join("");
  } catch (e) {
    $("answer").textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

refreshStats();
refreshDocs();
