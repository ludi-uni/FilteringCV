import { useState } from "react";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import {
  FieldEditor,
  formatClientIdLines,
  parseClientIdLines,
  parseNumberDraft,
  tryParseJsonArray,
} from "./ConfigFields";

afterEach(() => {
  cleanup();
});

function Harness({
  path,
  initial,
  original,
}: {
  path: string[];
  initial: unknown;
  original?: unknown;
}) {
  const [value, setValue] = useState(initial);
  return (
    <div>
      <FieldEditor
        path={path}
        value={value}
        original={original === undefined ? initial : original}
        onChange={(_path, next) => setValue(next)}
      />
      <pre data-testid="committed">{JSON.stringify(value)}</pre>
    </div>
  );
}

describe("parse helpers", () => {
  it("should_keep_empty_lines_out_of_parsed_client_ids", () => {
    expect(parseClientIdLines("a\n\nb\n")).toEqual(["a", "b"]);
    expect(formatClientIdLines(["a", "b"])).toBe("a\nb");
  });

  it("should_return_undefined_for_incomplete_json_arrays", () => {
    expect(tryParseJsonArray("[")).toBeUndefined();
    expect(tryParseJsonArray('["a",')).toBeUndefined();
    expect(tryParseJsonArray('["a"]')).toEqual(["a"]);
  });

  it("should_allow_partial_number_drafts_without_committing", () => {
    expect(parseNumberDraft("")).toBeNull();
    expect(parseNumberDraft("0.")).toBeUndefined();
    expect(parseNumberDraft("-")).toBeUndefined();
    expect(parseNumberDraft("0.068")).toBe(0.068);
    expect(parseNumberDraft("40")).toBe(40);
  });
});

describe("FieldEditor text input UX", () => {
  it("should_preserve_trailing_newline_while_editing_client_ids", async () => {
    const user = userEvent.setup();
    render(<Harness path={["speakers", "include_client_ids"]} initial={["abc"]} />);
    const box = screen.getByRole("textbox");
    await user.click(box);
    await user.keyboard("{End}{Enter}");
    expect((box as HTMLTextAreaElement).value).toBe("abc\n");
    expect(screen.getByTestId("committed").textContent).toBe('["abc"]');
  });

  it("should_keep_incomplete_json_array_text_visible", async () => {
    const user = userEvent.setup();
    render(<Harness path={["genders"]} initial={["female"]} />);
    const box = screen.getByRole("textbox");
    await user.clear(box);
    await user.type(box, "[[");
    expect((box as HTMLTextAreaElement).value).toBe("[");
    expect(screen.getByTestId("committed").textContent).toBe('["female"]');
  });

  it("should_keep_decimal_point_while_typing_numbers", async () => {
    const user = userEvent.setup();
    render(<Harness path={["quality_gate", "min_sec_per_mora"]} initial={0.068} />);
    const box = screen.getByRole("textbox");
    await user.clear(box);
    await user.type(box, "0.");
    expect((box as HTMLInputElement).value).toBe("0.");
  });

  it("should_keep_number_input_when_cleared_to_null", async () => {
    const user = userEvent.setup();
    render(<Harness path={["input", "max_clips"]} initial={10} />);
    const box = screen.getByRole("textbox");
    await user.clear(box);
    expect(screen.getByRole("textbox")).toBeTruthy();
    expect(screen.getByTestId("committed").textContent).toBe("null");
    expect(screen.queryByText("Set number")).toBeNull();
  });

  it("should_not_remount_string_input_when_crossing_80_chars", async () => {
    const user = userEvent.setup();
    const start = "a".repeat(79);
    render(<Harness path={["input", "corpus_root"]} initial={start} />);
    const box = screen.getByRole("textbox");
    await user.click(box);
    await user.keyboard("{End}xx");
    expect(document.activeElement).toBe(box);
    expect((box as HTMLTextAreaElement).value.length).toBe(81);
  });

  it("should_show_textbox_for_null_path_fields", () => {
    render(
      <Harness
        path={["mfa_gate", "work_dir"]}
        initial={null}
        original={null}
      />,
    );
    expect(screen.getByRole("textbox")).toBeTruthy();
    expect(screen.queryByText("Set string")).toBeNull();
  });

  it("should_edit_empty_object_arrays_as_json_text", async () => {
    const user = userEvent.setup();
    render(<Harness path={["audio_pipeline", "steps"]} initial={[]} />);
    const box = screen.getByRole("textbox");
    await user.clear(box);
    await user.type(box, "[[]");
    expect((box as HTMLTextAreaElement).value).toBe("[]");
    expect(screen.getByTestId("committed").textContent).toBe("[]");
  });
});
