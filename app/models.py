from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM structured-output schemas (what we ask the model to return)
# ---------------------------------------------------------------------------


class ExtractedFactLLM(BaseModel):
    """One atomic claim as returned by the fact-extraction LLM call."""

    subject: str
    predicate: str
    raw_value: str
    raw_unit: Optional[str] = None
    numeric_value: Optional[float] = None
    period_label: Optional[str] = None
    scope: Optional[str] = None
    status: Optional[str] = Field(
        default=None,
        description="actual | estimate | first_advance_estimate | second_advance_estimate | forecast | projection | target | historical",
    )
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    supporting_quote: str = Field(description="Verbatim substring of the source evidence that supports this fact")


class FactExtractionResult(BaseModel):
    facts: list[ExtractedFactLLM] = Field(default_factory=list)


class ChartExtractionLLM(BaseModel):
    """Result of sending a chart/figure crop to the vision LLM."""

    # Defaults to False rather than being required: a vision response describing why a page has
    # nothing to extract (e.g. a cover page) is still useful and shouldn't be discarded by a
    # schema-validation failure just because it omitted this one boolean - verified live, where
    # exactly this omission burned a full retry-and-fail cycle instead of just meaning "no data".
    has_extractable_data: bool = False
    title: Optional[str] = None
    chart_type: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    series: list[dict[str, Any]] = Field(default_factory=list)
    values_text: Optional[str] = Field(
        default=None, description="Plain-text rendering of the extracted data, e.g. as a small table"
    )
    unit: Optional[str] = None
    period: Optional[str] = None
    source_note: Optional[str] = None
    confidence: float = 0.0
    uncertainty_note: Optional[str] = None


class RelationshipClassificationLLM(BaseModel):
    relationship_type: Literal[
        "CORROBORATES", "CONTRADICTS", "LIKELY_CONTRADICTION", "CONTEXT_RECONCILES", "UNRELATED"
    ]
    confidence: float
    context_dimension: Optional[str] = Field(
        default=None, description="What explains the relationship, e.g. time | scope | unit | estimate_vintage | none"
    )
    reason: str


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------


class JobStatusOut(BaseModel):
    job_id: str
    document_id: str
    status: str
    stage: Optional[str] = None
    progress: int = 0
    total_pages: int = 0
    pages_processed: int = 0
    facts_extracted: int = 0
    relationships_found: int = 0
    error: Optional[str] = None


class UploadResponse(BaseModel):
    document_id: str
    job_id: str
    status: str = "queued"


class DocumentOut(BaseModel):
    id: str
    filename: str
    title: Optional[str] = None
    page_count: Optional[int] = None
    created_at: str
    latest_job_status: Optional[str] = None
    fact_count: int = 0
    relationship_count: int = 0


class EvidenceOut(BaseModel):
    id: str
    document_id: str
    document_filename: Optional[str] = None
    page_number: int
    evidence_type: str
    text: Optional[str] = None
    artifact_path: Optional[str] = None
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None


class FactOut(BaseModel):
    id: str
    document_id: str
    document_filename: Optional[str] = None
    subject: str
    predicate: str
    raw_value: Optional[str] = None
    raw_unit: Optional[str] = None
    numeric_value: Optional[float] = None
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None
    period_label: Optional[str] = None
    scope: Optional[str] = None
    status: Optional[str] = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    extraction_confidence: Optional[float] = None
    evidence: list[EvidenceOut] = Field(default_factory=list)


class RelationshipOut(BaseModel):
    id: str
    relationship_type: str
    confidence: Optional[float] = None
    context_dimension: Optional[str] = None
    reason: Optional[str] = None
    fact_a: FactOut
    fact_b: FactOut


class QueryRequest(BaseModel):
    question: str


class QueryFactRef(BaseModel):
    fact_id: str
    subject: str
    predicate: str
    value: Optional[str] = None
    period: Optional[str] = None
    scope: Optional[str] = None
    confidence: Optional[float] = None


class QueryEvidenceRef(BaseModel):
    document: str
    page: int
    text: Optional[str] = None
    evidence_id: str


class QueryRelationshipRef(BaseModel):
    type: str
    fact_a: str
    fact_b: str
    reason: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    facts: list[QueryFactRef] = Field(default_factory=list)
    evidence: list[QueryEvidenceRef] = Field(default_factory=list)
    relationships: list[QueryRelationshipRef] = Field(default_factory=list)
