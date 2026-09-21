import { describe, expect, it } from "vitest";

import { band, formatDelta, formatDuration, formatScore } from "./format";

describe("format", () => {
  it("uses the same bands as the LLM rubric anchors", () => {
    expect(band(95)).toBe("excellent");
    expect(band(75)).toBe("strong");
    expect(band(74.9)).toBe("adequate");
    expect(band(40)).toBe("weak");
    expect(band(10)).toBe("poor");
  });

  it("formats scores, deltas and durations", () => {
    expect(formatScore(null)).toBe("—");
    expect(formatScore(71.6)).toBe("72");
    expect(formatDelta(4.4)).toBe("+4");
    expect(formatDelta(-3)).toBe("-3");
    expect(formatDelta(null)).toBeNull();
    expect(formatDuration(75)).toBe("1:15");
  });
});
