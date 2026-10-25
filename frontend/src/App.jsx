import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function variablesOf(template) {
  return [...new Set([...(template || "").matchAll(/\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g)].map((m) => m[1]))];
}

export default function App() {
  const [workspace, setWorkspace] = useState(null);
  const [prompts, setPrompts] = useState([]);
  const [versions, setVersions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [title, setTitle] = useState("");
  const [template, setTemplate] = useState("Summarize for {{audience}}: {{article}}");
  const [values, setValues] = useState({});
  const [testInput, setTestInput] = useState("");
  const [group, setGroup] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState("");
  const [raw, setRaw] = useState("");
  const [structured, setStructured] = useState(null);
  const [engineGroup, setEngineGroup] = useState(null);
  const [engineModel, setEngineModel] = useState("openai/gpt-oss-20b");
  const [engineTestInput, setEngineTestInput] = useState("");
  const [engineBusy, setEngineBusy] = useState("");
  const [saving, setSaving] = useState(false);

  async function loadWorkspace() {
    let list = await api("/workspaces");
    if (!list.length) list = [await api("/workspaces", { method: "POST", body: JSON.stringify({ name: "Default" }) })];
    setWorkspace(list[0]);
    setPrompts(await api(`/workspaces/${list[0].id}/prompts`));
  }

  useEffect(() => { loadWorkspace().catch((e) => setError(e.message)); }, []);

  async function selectPrompt(prompt) {
    setSelected(prompt);
    setVersions(await api(`/prompts/${prompt.id}/versions`));
    setGroup(null);
    setAnalytics(null);
  }

  async function createPrompt() {
    // An empty title used to create an invisible Library button, so
    // users clicked Save repeatedly thinking nothing was created.
    const safeTitle = title.trim() || template.trim().split("\n")[0].slice(0, 60) || "Untitled prompt";
    setSaving(true);
    try {
      await api(`/workspaces/${workspace.id}/prompts`, {
        method: "POST", body: JSON.stringify({ title: safeTitle, template_text: template }),
      });
      setPrompts(await api(`/workspaces/${workspace.id}/prompts`));
    } catch (e) { setError(e.message); }
    setSaving(false);
  }

  async function structurePrompt() {
    setError("");
    setEngineBusy("structuring");
    try {
      setStructured(await api("/prompt-engine/structure", {
        method: "POST", body: JSON.stringify({ raw }),
      }));
      setEngineGroup(null);
    } catch (e) { setError(e.message); }
    setEngineBusy("");
  }

  async function runEngine() {
    setError("");
    setEngineBusy("running");
    try {
      const res = await api("/prompt-engine/run", {
        method: "POST",
        body: JSON.stringify({ raw, models: [engineModel], test_input: engineTestInput }),
      });
      setStructured(res.structured);
      setEngineGroup({ responses: res.responses });
    } catch (e) { setError(e.message); }
    setEngineBusy("");
  }

  async function saveStructured() {
    if (!structured) return;
    const safeTitle = title.trim() || structured.objective.slice(0, 60) || "Untitled prompt";
    setSaving(true);
    try {
      await api(`/workspaces/${workspace.id}/prompts`, {
        method: "POST", body: JSON.stringify({ title: safeTitle, template_text: structured.template_text }),
      });
      setPrompts(await api(`/workspaces/${workspace.id}/prompts`));
      setTemplate(structured.template_text);
    } catch (e) { setError(e.message); }
    setSaving(false);
  }

  async function runTest() {
    try {
      const version = versions[versions.length - 1];
      const res = await api("/runs", {
        method: "POST",
        body: JSON.stringify({ version_id: version.id, models: ["mock"], variables: values, test_input: testInput }),
      });
      setGroup(await api(`/run-groups/${res.group_id}`));
      setAnalytics(await api(`/run-groups/${res.group_id}/analytics`));
    } catch (e) { setError(e.message); }
  }

  const currentVersion = versions[versions.length - 1];

  return (
    <main className="max-w-5xl mx-auto p-6 space-y-8">
      <h1 className="text-3xl font-bold">OmniPrompt</h1>
      <p className="text-gray-600">Structure, version, compare, measure.</p>
      {error && <p className="text-red-600">{error}</p>}

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Prompt Engine — make a prompt from a raw request</h2>
        <p className="text-gray-600 text-sm">Type what you want, in plain words. Structure turns it into a testable prompt; Run actually runs it.</p>
        <textarea className="border p-2 w-full h-24" placeholder="e.g. Give me a Booth's multiplier example" value={raw} onChange={(e) => setRaw(e.target.value)} />
        <input className="border p-2 w-full" placeholder="Test input (optional)" value={engineTestInput} onChange={(e) => setEngineTestInput(e.target.value)} />
        <div className="flex gap-2 flex-wrap items-center">
          <select className="border p-2" value={engineModel} onChange={(e) => setEngineModel(e.target.value)}>
            <option value="mock">mock (free, instant)</option>
            <option value="openai/gpt-oss-20b">openai/gpt-oss-20b (NVIDIA, free, verified)</option>
            <option value="deepseek-ai/deepseek-v4.1-flash">deepseek-ai/deepseek-v4.1-flash (NVIDIA, slow)</option>
            <option value="moonshotai/kimi-k3">moonshotai/kimi-k3 (NVIDIA)</option>
            <option value="z-ai/glm-5.3">z-ai/glm-5.3 (NVIDIA, slow)</option>
          </select>
          <button className="bg-black text-white px-4 py-2 disabled:opacity-50" disabled={!raw.trim() || !!engineBusy} onClick={structurePrompt}>
            {engineBusy === "structuring" ? "Structuring…" : "Structure it"}
          </button>
          <button className="bg-black text-white px-4 py-2 disabled:opacity-50" disabled={!raw.trim() || !!engineBusy} onClick={runEngine}>
            {engineBusy === "running" ? "Running…" : "Run it"}
          </button>
        </div>
        {structured && (
          <div className="border p-4 space-y-2">
            <p><b>Role:</b> {structured.role} · <b>Audience:</b> {structured.audience || "—"} · <b>Format:</b> {structured.format_requirements || "—"}</p>
            {structured.gaps?.length > 0 && <p className="text-sm text-gray-600">Gaps (not invented): {structured.gaps.join(", ")}</p>}
            <pre className="bg-gray-100 p-4 text-sm whitespace-pre-wrap">{structured.template_text}</pre>
            <button className="bg-black text-white px-4 py-2 disabled:opacity-50" disabled={saving} onClick={saveStructured}>
              {saving ? "Saving…" : "Save this structured prompt to Library"}
            </button>
          </div>
        )}
        {engineGroup && engineGroup.responses.map((r, i) => (
          <article key={i} className="border p-4">
            <h3 className="font-semibold">{r.model}</h3>
            <pre className="whitespace-pre-wrap break-words text-sm">{r.error || r.output_text}</pre>
            <p className="text-sm text-gray-600">
              Score: {r.scores?.weighted_score ?? "—"} · Latency: {r.latency_ms}ms · Cost: ${r.cost_usd}
            </p>
          </article>
        ))}
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Prompt editor</h2>
        <input className="border p-2 w-full" placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
        <textarea className="border p-2 w-full h-28" value={template} onChange={(e) => setTemplate(e.target.value)} />
        <button className="bg-black text-white px-4 py-2 disabled:opacity-50" disabled={saving} onClick={createPrompt}>
          {saving ? "Saving…" : "Save as new prompt (v1)"}
        </button>
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Library</h2>
        {prompts.map((p) => (
          <button key={p.id} className="block underline" onClick={() => selectPrompt(p)}>{p.title || "(untitled)"}</button>
        ))}
        {selected && <p>Versions: {versions.map((v) => `v${v.version_no}`).join(", ")} — latest: {currentVersion?.template_text}</p>}
      </section>

      {currentVersion && (
        <section className="space-y-2">
          <h2 className="text-xl font-semibold">A/B sandbox</h2>
          {variablesOf(currentVersion.template_text).map((name) => (
            <input key={name} className="border p-2 w-full" placeholder={name}
              onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))} />
          ))}
          <input className="border p-2 w-full" placeholder="Test input" value={testInput} onChange={(e) => setTestInput(e.target.value)} />
          <button className="bg-black text-white px-4 py-2" onClick={runTest}>Run (mock provider)</button>
        </section>
      )}

      {group && (
        <section className="space-y-4">
          <h2 className="text-xl font-semibold">Results</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {group.responses.map((r, i) => (
              <article key={i} className="border p-4">
                <h3 className="font-semibold">{r.model}</h3>
                <pre className="whitespace-pre-wrap break-words text-sm">{r.error || r.output_text}</pre>
                <p className="text-sm text-gray-600">
                  Score: {r.scores?.weighted_score} · Latency: {r.latency_ms}ms · Cost: ${r.cost_usd}
                </p>
              </article>
            ))}
          </div>
          {analytics && <pre className="bg-gray-100 p-4 text-sm">{JSON.stringify(analytics, null, 2)}</pre>}
        </section>
      )}
    </main>
  );
}
