import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ScoreRing } from "./index";

describe("ScoreRing", () => {
  it("shows the score with its band label, not colour alone", () => {
    render(<ScoreRing score={82} />);
    expect(screen.getByText("82")).toBeInTheDocument();
    expect(screen.getByText("Strong")).toBeInTheDocument();
    expect(screen.getByText(/Score 82 out of 100, Strong/)).toBeInTheDocument();
  });

  it("handles a missing score", () => {
    render(<ScoreRing score={null} />);
    expect(screen.getByText("No score")).toBeInTheDocument();
  });
});
