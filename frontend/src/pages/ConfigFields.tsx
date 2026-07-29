import { useEffect, useRef, useState } from "react";

function pathsEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function useDraftText(external: string): {
  text: string;
  setText: (value: string) => void;
  focusProps: {
    onFocus: () => void;
    onBlur: () => void;
  };
} {
  const [text, setText] = useState(external);
  const focusedRef = useRef(false);

  useEffect(() => {
    if (!focusedRef.current) {
      setText(external);
    }
  }, [external]);

  return {
    text,
    setText,
    focusProps: {
      onFocus: () => {
        focusedRef.current = true;
      },
      onBlur: () => {
        focusedRef.current = false;
      },
    },
  };
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
  const { text, setText, focusProps } = useDraftText(external);

  return (
    <label className="config-field">
      <span className="config-field-label">
        {dotted}
        {dirty && <span className="dirty-dot" title="modified" />}
      </span>
      <span className="config-help">One client_id per line. Empty = all speakers.</span>
      <textarea
        className="textarea mono"
        rows={8}
        value={text}
        onFocus={focusProps.onFocus}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          onChange(path, parseClientIdLines(next));
        }}
        onBlur={() => {
          focusProps.onBlur();
          const ids = parseClientIdLines(text);
          setText(formatClientIdLines(ids));
          onChange(path, ids);
        }}
      />
    </label>
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
  const { text, setText, focusProps } = useDraftText(external);

  return (
    <label className="config-field">
      <span className="config-field-label">
        {dotted}
        {dirty && <span className="dirty-dot" title="modified" />}
      </span>
      <input
        className="input mono"
        inputMode="decimal"
        value={text}
        onFocus={focusProps.onFocus}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          const parsed = parseNumberDraft(next);
          if (parsed !== undefined) onChange(path, parsed);
        }}
        onBlur={() => {
          focusProps.onBlur();
          const parsed = parseNumberDraft(text);
          if (parsed === undefined) {
            setText(external);
            return;
          }
          onChange(path, parsed);
          setText(parsed == null ? "" : String(parsed));
        }}
      />
      <button
        type="button"
        className="btn btn-sm"
        style={{ marginTop: "0.35rem", width: "fit-content" }}
        onClick={() => onChange(path, null)}
      >
        Set null
      </button>
    </label>
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
  const { text, setText, focusProps } = useDraftText(external);

  return (
    <label className="config-field">
      <span className="config-field-label">
        {dotted}
        {dirty && <span className="dirty-dot" title="modified" />}
      </span>
      <span className="config-help">JSON array</span>
      <textarea
        className="textarea mono"
        rows={4}
        value={text}
        onFocus={focusProps.onFocus}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          const parsed = tryParseJsonArray(next);
          if (parsed !== undefined) onChange(path, parsed);
        }}
        onBlur={() => {
          focusProps.onBlur();
          const parsed = tryParseJsonArray(text);
          if (parsed !== undefined) {
            onChange(path, parsed);
            setText(JSON.stringify(parsed, null, 2));
            return;
          }
          setText(external);
        }}
      />
    </label>
  );
}

function StringFieldEditor({
  path,
  value,
  original,
  onChange,
}: {
  path: string[];
  value: string;
  original: unknown;
  onChange: (path: string[], value: unknown) => void;
}) {
  const label = path[path.length - 1] ?? "value";
  const dotted = path.join(".");
  const dirty = !pathsEqual(value, original);
  // Always use textarea so crossing length/newline thresholds does not remount
  // the control and steal focus/caret while typing.
  const tall =
    value.includes("\n") || value.length > 80 || label.endsWith("_ids");

  return (
    <label className="config-field">
      <span className="config-field-label">
        {dotted}
        {dirty && <span className="dirty-dot" title="modified" />}
      </span>
      <textarea
        className="textarea mono"
        rows={tall ? 4 : 2}
        value={value}
        onChange={(e) => onChange(path, e.target.value)}
      />
      {(original !== null && value !== "") || value === "" ? (
        <button
          type="button"
          className="btn btn-sm"
          style={{ marginTop: "0.35rem", width: "fit-content" }}
          onClick={() => onChange(path, null)}
        >
          Set null
        </button>
      ) : null}
    </label>
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

  if (dotted === "speakers.include_client_ids" && Array.isArray(value)) {
    return (
      <ClientIdListEditor path={path} value={value} original={original} onChange={onChange} />
    );
  }

  if (typeof value === "boolean") {
    return (
      <label className="config-field config-field-inline">
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(path, e.target.checked)}
        />
        <span className="config-field-label">
          {dotted}
          {dirty && <span className="dirty-dot" title="modified" />}
        </span>
      </label>
    );
  }

  if (typeof value === "number" || (value === null && typeof original === "number")) {
    return (
      <NumberFieldEditor
        path={path}
        value={typeof value === "number" ? value : null}
        original={original}
        onChange={onChange}
      />
    );
  }

  if (value === null) {
    return (
      <label className="config-field">
        <span className="config-field-label">
          {dotted} <span className="pill">null</span>
          {dirty && <span className="dirty-dot" title="modified" />}
        </span>
        <div className="toolbar">
          <button type="button" className="btn btn-sm" onClick={() => onChange(path, "")}>
            Set string
          </button>
          <button type="button" className="btn btn-sm" onClick={() => onChange(path, 0)}>
            Set number
          </button>
          <button type="button" className="btn btn-sm" onClick={() => onChange(path, false)}>
            Set boolean
          </button>
        </div>
      </label>
    );
  }

  if (typeof value === "string") {
    return <StringFieldEditor path={path} value={value} original={original} onChange={onChange} />;
  }

  if (Array.isArray(value)) {
    const allPrimitive = value.every(
      (item) => item == null || ["string", "number", "boolean"].includes(typeof item),
    );
    if (allPrimitive) {
      return <JsonArrayEditor path={path} value={value} original={original} onChange={onChange} />;
    }
  }

  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <fieldset className="config-group">
        <legend>
          {dotted || label}
          {dirty && <span className="dirty-dot" title="modified" />}
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
    <label className="config-field">
      <span className="config-field-label">{dotted}</span>
      <input className="input mono" value={String(value)} readOnly />
    </label>
  );
}
