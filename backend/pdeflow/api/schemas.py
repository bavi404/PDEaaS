"""Request/response models. Field names keep the two targets separate everywhere:
`official_pdebench_target` (PRIMARY benchmark) vs `independent_steady_state_reference` (SECONDARY, analysis)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["fno_native", "fno64_interpolated", "fno128_native"]
Grid = list[list[float]]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    permeability: Grid | None = Field(
        None, description="Square 64x64 or 128x128 permeability field a(x,y) > 0, indexed [x][y] on cell centres "
                          "(i+0.5)/n. Training data are binary {0.1, 1.0}.")
    sample_id: str | None = Field(None, description="Id of a bundled test sample (see GET /samples).")
    resolution: Literal[64, 128] = Field(128, description="Only with sample_id: resolution to evaluate at.")
    strategy: Strategy | None = Field(
        None, description="Inference path. Default: fno_native at 64x64; fno64_interpolated at 128x128 "
                          "(lower measured test error than fno128_native, docs/RESULTS.md).")
    return_targets: bool = Field(False, description="Only with sample_id: also return both targets.")

    @model_validator(mode="after")
    def _one_input(self) -> PredictRequest:
        if (self.permeability is None) == (self.sample_id is None):
            raise ValueError("provide exactly one of 'permeability' or 'sample_id'")
        if self.permeability is not None and self.return_targets:
            raise ValueError("'return_targets' requires 'sample_id' (no targets exist for user input)")
        return self


class ErrorMetric(BaseModel):
    relative_l2: float


class SampleEvaluation(BaseModel):
    """Per-sample error of this prediction. The two targets are reported separately and never combined."""
    primary_vs_official_pdebench_target: ErrorMetric
    secondary_vs_independent_steady_state_reference: ErrorMetric
    official_target_vs_reference: ErrorMetric


class Targets(BaseModel):
    official_pdebench_target: Grid
    independent_steady_state_reference: Grid


class PredictResponse(BaseModel):
    prediction: Grid
    resolution: int
    model_resolution: int
    strategy: Strategy
    inference_ms: float
    checkpoint_sha256: str
    warnings: list[str]
    input_summary: dict
    sample_id: str | None = None
    evaluation: SampleEvaluation | None = None
    targets: Targets | None = None


class SampleSummary(BaseModel):
    id: str
    test_index: int
    fraction_high_permeability: float
    official_target_pde_residual: float


class SampleDetail(SampleSummary):
    resolution: int
    permeability: Grid
    targets: Targets


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    checkpoint_sha256: str
    n_samples: int
