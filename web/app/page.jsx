"use client";

import { useEffect, useMemo, useState } from "react";

async function fetchJson(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return await res.json();
}

function modelGroupKey(m) {
  return m.group ?? m.provider_label ?? m.provider_id ?? "Other";
}

function loadStringArray(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return parsed.map((v) => String(v));
  } catch {
    return null;
  }
}

function saveStringArray(key, arr) {
  try {
    localStorage.setItem(key, JSON.stringify(arr));
  } catch {
    // ignore
  }
}

export default function Page() {
  const [words, setWords] = useState([]);
  const [models, setModels] = useState([]);
  const [clips, setClips] = useState([]);
  const [failures, setFailures] = useState([]);
  const [query, setQuery] = useState("");
  const [enabledGroups, setEnabledGroups] = useState(() => new Set());
  const [enabledKinds, setEnabledKinds] = useState(() => new Set());
  const [groupToAdd, setGroupToAdd] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const [wordsJson, modelsJson, clipsJson] = await Promise.all([
          fetchJson("data/words.json"),
          fetchJson("data/models.json"),
          fetchJson("data/clips.json").catch(() => ({ clips: [] }))
        ]);
        const nextWords = wordsJson.words ?? [];
        const nextModels = modelsJson.models ?? [];
        const nextClips = clipsJson.clips ?? [];

        setWords(nextWords);
        setModels(nextModels);
        setClips(nextClips);
        setFailures(clipsJson.failures ?? []);

        const modelIdsWithClips = new Set(nextClips.map((c) => c.model_id));
        const generatedModels = nextModels.filter((m) => modelIdsWithClips.has(m.id));
        const groups = Array.from(new Set(generatedModels.map(modelGroupKey))).sort((a, b) => a.localeCompare(b));
        const kinds = Array.from(new Set(generatedModels.map((m) => m.input_kind))).sort((a, b) => a.localeCompare(b));

        const storedGroups = loadStringArray("pv_enabledGroups_v1");
        const storedKinds = loadStringArray("pv_enabledKinds_v1");

        const nextEnabledGroups =
          storedGroups != null ? storedGroups.filter((g) => groups.includes(g)) : groups.length <= 6 ? groups : [];
        const nextEnabledKinds = storedKinds != null ? storedKinds.filter((k) => kinds.includes(k)) : kinds;

        setEnabledGroups(new Set(nextEnabledGroups));
        setEnabledKinds(new Set(nextEnabledKinds));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const modelIdsWithClips = useMemo(() => new Set(clips.map((c) => c.model_id)), [clips]);

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

  const generatedModels = useMemo(() => models.filter((m) => modelIdsWithClips.has(m.id)), [models, modelIdsWithClips]);

  const filteredWords = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return words;
    return words.filter((w) => {
      const hay = `${w.id} ${w.fa} ${w.ar ?? ""} ${w.fa_diac ?? ""} ${w.latn ?? ""} ${w.gloss_en ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [words, query]);

  const allGroups = useMemo(() => {
    const groups = Array.from(new Set(generatedModels.map(modelGroupKey)));
    groups.sort((a, b) => a.localeCompare(b));
    return groups;
  }, [generatedModels]);

  const allKinds = useMemo(() => {
    const kinds = Array.from(new Set(generatedModels.map((m) => m.input_kind)));
    kinds.sort((a, b) => a.localeCompare(b));
    return kinds;
  }, [generatedModels]);

  const visibleModels = useMemo(() => {
    return generatedModels
      .filter((m) => enabledGroups.has(modelGroupKey(m)))
      .filter((m) => enabledKinds.has(m.input_kind))
      .sort((a, b) => {
        const ga = modelGroupKey(a);
        const gb = modelGroupKey(b);
        if (ga !== gb) return ga.localeCompare(gb);
        return String(a.input_kind).localeCompare(String(b.input_kind));
      });
  }, [generatedModels, enabledGroups, enabledKinds]);

  const enabledGroupList = useMemo(() => allGroups.filter((g) => enabledGroups.has(g)), [allGroups, enabledGroups]);
  const disabledGroupList = useMemo(() => allGroups.filter((g) => !enabledGroups.has(g)), [allGroups, enabledGroups]);

  useEffect(() => {
    saveStringArray("pv_enabledGroups_v1", Array.from(enabledGroups));
  }, [enabledGroups]);

  useEffect(() => {
    saveStringArray("pv_enabledKinds_v1", Array.from(enabledKinds));
  }, [enabledKinds]);

  function disableGroup(group) {
    setEnabledGroups((prev) => {
      const next = new Set(prev);
      next.delete(group);
      return next;
    });
  }

  function enableGroup(group) {
    if (!group || !allGroups.includes(group)) return;
    setEnabledGroups((prev) => new Set(prev).add(group));
    // Auto-enable all text variants when adding a model
    setEnabledKinds(new Set(allKinds));
  }

  function enableAllGroups() {
    setEnabledGroups(new Set(allGroups));
    // Auto-enable all text variants when enabling models
    setEnabledKinds(new Set(allKinds));
  }

  function disableAllGroups() {
    setEnabledGroups(new Set());
  }

  function toggleKind(kind) {
    setEnabledKinds((prev) => {
      const next = new Set(prev);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
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
            {words.length} words · {generatedModels.length} generated model-variants · {clips.length} clips
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
          <span className="subtitle">Enabled models: {enabledGroupList.length}</span>
          <button className="button" type="button" onClick={enableAllGroups} disabled={enabledGroupList.length === allGroups.length}>
            Enable all
          </button>
          <button className="button" type="button" onClick={disableAllGroups} disabled={enabledGroupList.length === 0}>
            Disable all
          </button>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <span className="subtitle">Text variants:</span>
          {allKinds.map((kind) => (
            <label key={kind} className="kindToggle">
              <input type="checkbox" checked={enabledKinds.has(kind)} onChange={() => toggleKind(kind)} />
              <span>{kind}</span>
            </label>
          ))}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <span className="subtitle">Enable model:</span>
          <input
            className="input"
            style={{ minWidth: 360 }}
            list="group-options"
            placeholder={disabledGroupList.length ? "Start typing a model name…" : "No disabled models"}
            value={groupToAdd}
            onChange={(e) => setGroupToAdd(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                enableGroup(groupToAdd.trim());
                setGroupToAdd("");
              }
            }}
            disabled={!disabledGroupList.length}
          />
          <datalist id="group-options">
            {disabledGroupList.map((g) => (
              <option key={g} value={g} />
            ))}
          </datalist>
          <button
            className="button"
            type="button"
            disabled={!disabledGroupList.length || !groupToAdd.trim()}
            onClick={() => {
              enableGroup(groupToAdd.trim());
              setGroupToAdd("");
            }}
          >
            Add
          </button>
        </div>
        <div className="chips" style={{ marginTop: 10 }}>
          {enabledGroupList.length === 0 ? (
            <div className="missing">No models enabled. Use “Enable model” above to add one.</div>
          ) : null}
          {enabledGroupList.map((group) => (
            <div key={group} className="chip" title={group}>
              <span className="chipLabel">{group}</span>
              <button className="chipRemove" type="button" onClick={() => disableGroup(group)} aria-label={`Disable ${group}`}>
                ×
              </button>
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
