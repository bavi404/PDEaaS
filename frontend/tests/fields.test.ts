import { describe, expect, it } from "vitest";
import { absDiff, distributionWarnings, generateField, gridRange, parseField, validateGrid } from "@/lib/fields";
import { grid } from "./fixtures";

const binary64 = grid(64, (i, j) => ((i * 7 + j) % 5 < 2 ? 1 : 0.1));

describe("field parsing and validation (mirrors the API)", () => {
  it("accepts a binary 64x64 JSON grid without warnings", () => {
    const r = parseField(JSON.stringify(binary64));
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.resolution).toBe(64);
      expect(r.warnings).toEqual([]);
    }
  });

  it("accepts CSV / whitespace rows and comments", () => {
    const csv = "# header\n" + grid(128, () => 0.1).map((r) => r.join(", ")).join("\n");
    const r = parseField(csv);
    expect(r.ok && r.resolution).toBe(128);
  });

  it.each([
    ["", "No data provided"],
    ["[[1, 2], [3", "Invalid JSON"],
    [JSON.stringify(grid(10, () => 1)), "not supported"],
    [JSON.stringify([...grid(64, () => 1).slice(0, 63), new Array(63).fill(1)]), "must be square"],
    [JSON.stringify(grid(64, () => 1)).replace("1", '"abc"'), "non-numeric"],
    [JSON.stringify(grid(64, (i) => (i === 3 ? -1 : 1))), "strictly positive"],
    [JSON.stringify(grid(64, (i) => (i === 3 ? 0 : 1))), "strictly positive"],
    [JSON.stringify(grid(64, (i) => (i === 3 ? 1e7 : 1))), "rejected"],
    ['{"a": 1}', "Expected a 2-D array"],
    ["[1, 2, 3]", "Every row"],
  ])("rejects malformed input %#", (text, msg) => {
    const r = parseField(text);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toContain(msg);
  });

  it("flags out-of-distribution values with the API's wording", () => {
    expect(distributionWarnings(grid(64, () => 0.5))).toEqual(["non-binary field: training data contain only the values 0.1 and 1.0; accuracy unknown"]);
    const w = distributionWarnings(grid(64, (i) => (i % 2 ? 5 : 0.1)));
    expect(w[0]).toMatch(/outside the training range \[0.1, 1\]/);
    expect(w).toHaveLength(2);
  });

  it("validateGrid rejects non-arrays", () => {
    expect(validateGrid(null).ok).toBe(false);
    expect(validateGrid([]).ok).toBe(false);
  });
});

describe("two-phase generator", () => {
  const p = { resolution: 64 as const, seed: 3, lengthScale: 0.08, highFraction: 0.4, low: 0.1, high: 1.0 };

  it("is deterministic and binary with the requested fraction", () => {
    const a = generateField(p);
    const b = generateField(p);
    expect(a).toEqual(b);
    const flat = a.flat();
    expect(new Set(flat)).toEqual(new Set([0.1, 1.0]));
    const frac = flat.filter((v) => v === 1).length / flat.length;
    expect(frac).toBeGreaterThan(0.38);
    expect(frac).toBeLessThan(0.42);
    expect(generateField({ ...p, seed: 4 })).not.toEqual(a);
  });

  it("produces the requested resolution and passes validation", () => {
    const g = generateField({ ...p, resolution: 128 });
    expect(g).toHaveLength(128);
    const r = validateGrid(g);
    expect(r.ok && r.warnings).toEqual([]);
  });

  it("handles 0% and 100% high fraction", () => {
    expect(new Set(generateField({ ...p, highFraction: 0 }).flat())).toEqual(new Set([0.1]));
    expect(new Set(generateField({ ...p, highFraction: 1 }).flat())).toEqual(new Set([1.0]));
  });
});

describe("grid helpers", () => {
  it("absDiff and gridRange", () => {
    const a = [[1, 2], [3, 4]];
    const b = [[1, 3], [1, 4]];
    expect(absDiff(a, b)).toEqual([[0, 1], [2, 0]]);
    expect(gridRange(a, b)).toEqual([1, 4]);
  });
});
