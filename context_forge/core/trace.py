"""
ContextForge Observability & Tracing Engine.
Captures microsecond-level timing, intermediate states, rank mutations,
and token metrics across the entire RAG lifecycle.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    name: str
    duration_ms: float
    input_count: int
    output_count: int
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class ExecutionTrace(BaseModel):
    pipeline_id: str
    query: str
    total_duration_ms: float = 0.0
    steps: List[TraceStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_step(
        self,
        name: str,
        duration_ms: float,
        input_count: int,
        output_count: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps.append(
            TraceStep(
                name=name,
                duration_ms=round(duration_ms, 2),
                input_count=input_count,
                output_count=output_count,
                details=details or {},
            )
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_duration_ms": self.total_duration_ms,
            "step_count": len(self.steps),
            "step_latencies": {s.name: s.duration_ms for s in self.steps},
        }


class TraceContext:
    """Context manager for timing pipeline steps."""

    def __init__(self, trace: ExecutionTrace, name: str, input_count: int = 0):
        self.trace = trace
        self.name = name
        self.input_count = input_count
        self.output_count = 0
        self.details: Dict[str, Any] = {}
        self._start_time = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def set_output(self, count: int, **details):
        self.output_count = count
        self.details.update(details)

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.trace.add_step(
            name=self.name,
            duration_ms=duration_ms,
            input_count=self.input_count,
            output_count=self.output_count,
            details=self.details,
        )
