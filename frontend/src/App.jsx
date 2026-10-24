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
    try {
      await api(`/workspaces/${workspace.id}/prompts`, {
        method: "POST", body: JSON.stringify({ title, template_text: template }),
      });
      setPrompts(await api(`/workspaces/${workspace.id}/prompts`));
    } catch (e) { setError(e.message); }
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
        <h2 className="text-xl font-semibold">Prompt editor</h2>
        <input className="border p-2 w-full" placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
        <textarea className="border p-2 w-full h-28" value={template} onChange={(e) => setTemplate(e.target.value)} />
        <button className="bg-black text-white px-4 py-2" onClick={createPrompt}>Save as new prompt (v1)</button>
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Library</h2>
        {prompts.map((p) => (
          <button key={p.id} className="block underline" onClick={() => selectPrompt(p)}>{p.title}</button>
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
                <p>{r.error || r.output_text}</p>
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
