import { useEffect, useId, useRef, useState, type RefObject } from "react";

function pathsEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function parseClientIdLines(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function formatClientIdLines(ids: unknown[]): string {
  return ids.map(String).join("\n");
}

export function tryParseJsonArray(text: string): unknown[] | undefined {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (Array.isArray(parsed)) return parsed;
  } catch {
    // incomplete JSON while typing
  }
  return undefined;
}

export function parseNumberDraft(text: string): number | null | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  if (trimmed === "-" || trimmed === "." || trimmed === "-.") return undefined;
  if (!/^-?\d*\.?\d*$/.test(trimmed)) return undefined;
  if (trimmed.endsWith(".")) return undefined;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : undefined;
}

function looksLikeStringField(path: string[]): boolean {
  const leaf = path[path.length - 1] ?? "";
  return (
    leaf.endsWith("_path") ||
    leaf.endsWith("_dir") ||
    leaf.endsWith("_root") ||
    leaf.endsWith("_id") ||
    leaf.endsWith("_ids") ||
    leaf.endsWith("_file") ||
    leaf.endsWith("_uri") ||
    leaf.endsWith("_url") ||
    leaf.endsWith("_name") ||
    leaf.endsWith("_text") ||
    leaf === "device" ||
    leaf === "method" ||
    leaf === "locale_expected" ||
    leaf === "clip_tsv" ||
    leaf === "audio_subdir" ||
    leaf === "hf_repo_id"
  );
}

/**
 * Uncontrolled text buffer that never fights the caret while focused.
 * Parent commits happen from the DOM value; prop sync only when blurred.
 */
function useUncontrolledText(external: string) {
  const ref = useRef<HTMLTextAreaElement | HTMLInputElement | null>(null);
  const focusedRef = useRef(false);
  const externalRef = useRef(external);

  useEffect(() => {
    externalRef.current = external;
    const el = ref.current;
    if (!el || focusedRef.current) return;
    if (el.value !== external) {
      el.value = external;
    }
  }, [external]);

  return {
    ref,
    defaultValue: external,
    onFocus: () => {
      focusedRef.current = true;
    },
    onBlur: () => {
      focusedRef.current = false;
    },
    readValue: () => ref.current?.value ?? externalRef.current,
    writeValue: (next: string) => {
      if (ref.current) ref.current.value = next;
    },
  };
}

function DirtyMark({ dirty }: { dirty: boolean }) {
  if (!dirty) return null;
  return <span className="dirty-dot" title="modified" />;
}

function ClientIdListEditor({
  path,
  value,
  original,
  onChange,
}: {
  path: string[];
  value: unknown[];
  original: unknown;
  onChange: (path: string[], value: unknown) => void;
}) {
  const dotted = path.join(".");
  const dirty = !pathsEqual(value, original);
  const external = formatClientIdLines(value);
  const draft = useUncontrolledText(external);
  const inputId = useId();

  return (
    <div className="config-field">
      <label className="config-field-label" htmlFor={inputId}>
        {dotted}
        <DirtyMark dirty={dirty} />
      </label>
      <span className="config-help">One client_id per line. Empty = all speakers.</span>
      <textarea
        id={inputId}
        ref={draft.ref as RefObject<HTMLTextAreaElement>}
        className="textarea mono"
        rows={8}
        defaultValue={draft.defaultValue}
        onFocus={draft.onFocus}
        onChange={(e) => onChange(path, parseClientIdLines(e.target.value))}
        onBlur={(e) => {
          draft.onBlur();
          const ids = parseClientIdLines(e.target.value);
          draft.writeValue(formatClientIdLines(ids));
          onChange(path, ids);
        }}
      />
    </div>
  );
}

function NumberFieldEditor({
  path,
  value,
  original,
  onChange,
}: {
  path: string[];
  value: number | null;
  original: unknown;
  onChange: (path: string[], value: unknown) => void;
}) {
  const dotted = path.join(".");
  const dirty = !pathsEqual(value, original);
  const external = value == null || !Number.isFinite(value) ? "" : String(value);
  const draft = useUncontrolledText(external);
  const inputId = useId();

  return (
    <div className="config-field">
      <label className="config-field-label" htmlFor={inputId}>
        {dotted}
        <DirtyMark dirty={dirty} />
      </label>
      <input
        id={inputId}
        ref={draft.ref as RefObject<HTMLInputElement>}
        className="input mono"
        inputMode="decimal"
        defaultValue={draft.defaultValue}
        onFocus={draft.onFocus}
        onChange={(e) => {
          const parsed = parseNumberDraft(e.target.value);
          if (parsed !== undefined) onChange(path, parsed);
        }}
        onBlur={(e) => {
          draft.onBlur();
          const parsed = parseNumberDraft(e.target.value);
          if (parsed === undefined) {
            draft.writeValue(external);
            return;
          }
          onChange(path, parsed);
          draft.writeValue(parsed == null ? "" : String(parsed));
        }}
      />
      <button
        type="button"
        className="btn btn-sm"
        style={{ width: "fit-content" }}
        onClick={() => {
          draft.writeValue("");
          onChange(path, null);
        }}
      >
        Set null
      </button>
    </div>
  );
}

function JsonArrayEditor({
  path,
  value,
  original,
  onChange,
}: {
  path: string[];
  value: unknown[];
  original: unknown;
  onChange: (path: string[], value: unknown) => void;
}) {
  const dotted = path.join(".");
  const dirty = !pathsEqual(value, original);
  const external = JSON.stringify(value, null, 2);
  const draft = useUncontrolledText(external);
  const inputId = useId();

  return (
    <div className="config-field">
      <label className="config-field-label" htmlFor={inputId}>
        {dotted}
        <DirtyMark dirty={dirty} />
      </label>
      <span className="config-help">JSON array</span>
      <textarea
        id={inputId}
        ref={draft.ref as RefObject<HTMLTextAreaElement>}
        className="textarea mono"
        rows={4}
        defaultValue={draft.defaultValue}
        onFocus={draft.onFocus}
        onChange={(e) => {
          const parsed = tryParseJsonArray(e.target.value);
          if (parsed !== undefined) onChange(path, parsed);
        }}
        onBlur={(e) => {
          draft.onBlur();
          const parsed = tryParseJsonArray(e.target.value);
          if (parsed !== undefined) {
            onChange(path, parsed);
            draft.writeValue(JSON.stringify(parsed, null, 2));
            return;
          }
          draft.writeValue(external);
        }}
      />
    </div>
  );
}

function StringFieldEditor({
  path,
  value,
  original,
  onChange,
  allowEmptyNull = false,
}: {
  path: string[];
  value: string;
  original: unknown;
  onChange: (path: string[], value: unknown) => void;
  allowEmptyNull?: boolean;
}) {
  const label = path[path.length - 1] ?? "value";
  const dotted = path.join(".");
  const dirty = !pathsEqual(value, original);
  const tall =
    value.includes("\n") || value.length > 80 || label.endsWith("_ids");
  const draft = useUncontrolledText(value);
  const inputId = useId();

  return (
    <div className="config-field">
      <label className="config-field-label" htmlFor={inputId}>
        {dotted}
        {allowEmptyNull && value === "" && original === null ? (
          <span className="pill">was null</span>
        ) : null}
        <DirtyMark dirty={dirty} />
      </label>
      <textarea
        id={inputId}
        ref={draft.ref as RefObject<HTMLTextAreaElement>}
        className="textarea mono"
        rows={tall ? 4 : 2}
        defaultValue={draft.defaultValue}
        onFocus={draft.onFocus}
        onChange={(e) => {
          const next = e.target.value;
          if (allowEmptyNull && next === "") {
            onChange(path, null);
            return;
          }
          onChange(path, next);
        }}
        onBlur={(e) => {
          draft.onBlur();
          const next = e.target.value;
          if (allowEmptyNull && next.trim() === "") {
            draft.writeValue("");
            onChange(path, null);
            return;
          }
          onChange(path, next);
        }}
      />
      <button
        type="button"
        className="btn btn-sm"
        style={{ width: "fit-content" }}
        onClick={() => {
          draft.writeValue("");
          onChange(path, null);
        }}
      >
        Set null
      </button>
    </div>
  );
}

export function FieldEditor({
  path,
  value,
  original,
  onChange,
}: {
  path: string[];
  value: unknown;
  original: unknown;
  onChange: (path: string[], value: unknown) => void;
}) {
  const label = path[path.length - 1] ?? "value";
  const dotted = path.join(".");
  const dirty = !pathsEqual(value, original);
  const [forceNumber, setForceNumber] = useState(false);
  const [forceString, setForceString] = useState(false);

  if (dotted === "speakers.include_client_ids" && Array.isArray(value)) {
    return (
      <ClientIdListEditor path={path} value={value} original={original} onChange={onChange} />
    );
  }

  if (typeof value === "boolean") {
    const inputId = `${dotted}-bool`;
    return (
      <div className="config-field config-field-inline">
        <input
          id={inputId}
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(path, e.target.checked)}
        />
        <label className="config-field-label" htmlFor={inputId}>
          {dotted}
          <DirtyMark dirty={dirty} />
        </label>
      </div>
    );
  }

  if (
    typeof value === "number" ||
    forceNumber ||
    (value === null && typeof original === "number")
  ) {
    return (
      <NumberFieldEditor
        path={path}
        value={typeof value === "number" ? value : null}
        original={original}
        onChange={(p, v) => {
          if (v === null) setForceNumber(true);
          onChange(p, v);
        }}
      />
    );
  }

  if (
    typeof value === "string" ||
    forceString ||
    (value === null && (typeof original === "string" || looksLikeStringField(path)))
  ) {
    return (
      <StringFieldEditor
        path={path}
        value={typeof value === "string" ? value : ""}
        original={original}
        allowEmptyNull={value === null || original === null || forceString}
        onChange={(p, v) => {
          if (typeof v === "string") setForceString(true);
          onChange(p, v);
        }}
      />
    );
  }

  if (value === null) {
    return (
      <div className="config-field">
        <span className="config-field-label">
          {dotted} <span className="pill">null</span>
          <DirtyMark dirty={dirty} />
        </span>
        <div className="toolbar">
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              setForceString(true);
              onChange(path, "");
            }}
          >
            Set string
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              setForceNumber(true);
              onChange(path, 0);
            }}
          >
            Set number
          </button>
          <button type="button" className="btn btn-sm" onClick={() => onChange(path, false)}>
            Set boolean
          </button>
        </div>
      </div>
    );
  }

  if (Array.isArray(value)) {
    // Always edit arrays as JSON text so empty arrays and object arrays remain typable.
    return <JsonArrayEditor path={path} value={value} original={original} onChange={onChange} />;
  }

  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <fieldset className="config-group">
        <legend>
          {dotted || label}
          <DirtyMark dirty={dirty} />
        </legend>
        {entries.map(([key, child]) => (
          <FieldEditor
            key={`${dotted}.${key}`}
            path={[...path, key]}
            value={child}
            original={
              original && typeof original === "object" && !Array.isArray(original)
                ? (original as Record<string, unknown>)[key]
                : undefined
            }
            onChange={onChange}
          />
        ))}
      </fieldset>
    );
  }

  return (
    <div className="config-field">
      <span className="config-field-label">{dotted}</span>
      <input className="input mono" value={String(value)} readOnly />
    </div>
  );
}
