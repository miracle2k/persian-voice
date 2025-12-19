"use client";

import { useEffect, useMemo, useState } from "react";

async function fetchJson(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return await res.json();
}

export default function Page() {
  const [words, setWords] = useState([]);
  const [models, setModels] = useState([]);
  const [clips, setClips] = useState([]);
  const [failures, setFailures] = useState([]);
  const [query, setQuery] = useState("");
  const [visibleModelIds, setVisibleModelIds] = useState(() => new Set());
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const [wordsJson, modelsJson, clipsJson] = await Promise.all([
          fetchJson("data/words.json"),
          fetchJson("data/models.json"),
          fetchJson("data/clips.json").catch(() => ({ clips: [] }))
        ]);
        setWords(wordsJson.words ?? []);
        setModels(modelsJson.models ?? []);
        setClips(clipsJson.clips ?? []);
        setFailures(clipsJson.failures ?? []);
        const defaultVisible = new Set((modelsJson.models ?? []).map((m) => m.id));
        setVisibleModelIds(defaultVisible);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const clipsByKey = useMemo(() => {
    const map = new Map();
    for (const c of clips) map.set(`${c.word_id}||${c.model_id}`, c);
    return map;
  }, [clips]);

  const failuresByKey = useMemo(() => {
    const map = new Map();
    for (const f of failures) map.set(`${f.word_id}||${f.model_id}`, f);
    return map;
  }, [failures]);

  const filteredWords = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return words;
    return words.filter((w) => {
      const hay = `${w.id} ${w.fa} ${w.ar ?? ""} ${w.fa_diac ?? ""} ${w.latn ?? ""} ${w.gloss_en ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [words, query]);

  const visibleModels = useMemo(() => models.filter((m) => visibleModelIds.has(m.id)), [models, visibleModelIds]);

  const modelGroups = useMemo(() => {
    const groups = new Map();
    for (const m of models) {
      const key = m.group ?? m.provider_label ?? m.provider_id ?? "Other";
      const arr = groups.get(key) ?? [];
      arr.push(m);
      groups.set(key, arr);
    }
    return Array.from(groups.entries()).map(([group, list]) => [group, list.sort((a, b) => a.label.localeCompare(b.label))]);
  }, [models]);

  function toggleModel(id) {
    setVisibleModelIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  if (error) {
    return (
      <div className="container">
        <h1 className="title">Persian Voice Comparison</h1>
        <p className="subtitle">Failed to load dataset.</p>
        <pre className="panel">{error}</pre>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="header">
        <div>
          <h1 className="title">Persian Voice Comparison</h1>
          <p className="subtitle">
            {words.length} words · {models.length} model-variants · {clips.length} clips
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="row">
          <input
            className="input"
            placeholder="Filter words (id / فارسی / translit / gloss)…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="subtitle">Columns:</span>
        </div>
        <div className="chips" style={{ marginTop: 10 }}>
          {modelGroups.map(([group, groupModels]) => (
            <div key={group} className="chip" title={group}>
              <span style={{ fontWeight: 600 }}>{group}</span>
              <span style={{ color: "var(--muted)" }}>·</span>
              {groupModels.map((m) => (
                <label key={m.id} style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                  <input type="checkbox" checked={visibleModelIds.has(m.id)} onChange={() => toggleModel(m.id)} />
                  <span style={{ whiteSpace: "nowrap" }}>{m.input_kind}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th style={{ minWidth: 240 }}>Word</th>
              {visibleModels.map((m) => (
                <th key={m.id} style={{ minWidth: 240 }}>
                  <div style={{ fontWeight: 700, color: "var(--fg)" }}>{m.provider_label}</div>
                  <div>{m.engine_id}</div>
                  <div>{m.voice_id}</div>
                  <div style={{ color: "var(--accent)" }}>{m.input_kind}</div>
                  {!m.available ? <div className="missing">{m.unavailable_reason ?? "Unavailable"}</div> : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredWords.map((w) => (
              <tr key={w.id}>
                <td>
                  <div className="fa">{w.fa}</div>
                  <div className="latn">
                    {w.latn}
                    {w.gloss_en ? ` · ${w.gloss_en}` : ""}
                  </div>
                </td>
                {visibleModels.map((m) => {
                  const clip = clipsByKey.get(`${w.id}||${m.id}`);
                  if (!clip) {
                    const failure = failuresByKey.get(`${w.id}||${m.id}`);
                    if (failure) {
                      return (
                        <td key={m.id} className="missing" title={`${failure.error_type}: ${failure.error}`}>
                          ERR: {failure.error_type}
                        </td>
                      );
                    }
                    return (
                      <td key={m.id} className="missing">
                        —
                      </td>
                    );
                  }
                  return (
                    <td key={m.id}>
                      <audio controls preload="none" src={clip.audio_path} />
                      <div className="latn" title={clip.text}>
                        {clip.text}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
