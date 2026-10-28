"""Environment configuration shared by the CLI, migrations and worker."""

from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KIP_", env_file=".env", extra="ignore")

    database_url: SecretStr = SecretStr("postgresql+psycopg://kip:kip_dev_only@127.0.0.1:5432/kip")
    storage_root: Path = Path("data/objects")
    lease_seconds: int = Field(default=300, ge=15)
    poll_seconds: float = Field(default=2.0, gt=0)
    parse_timeout_seconds: int = Field(default=120, ge=1)
    parse_memory_mb: int = Field(default=1024, ge=128)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: int = Field(default=30, ge=1)
    embed_batch_size: int = Field(default=32, ge=1, le=256)
    embed_device: str = "cpu"
    # Dense search over-fetches so that candidates removed by the PostgreSQL re-check
    # (deleted, superseded, unauthorized) do not silently shrink the result list.
    search_overfetch: int = Field(default=4, ge=1, le=20)
    # Replenishment doubles the per-retriever limit until enough candidates survive the
    # re-check, the retriever runs out, or this hard cap is reached.
    search_max_candidates: int = Field(default=200, ge=10, le=1000)
    # Hybrid: visible candidates wanted from each retriever before fusion (design: 50).
    hybrid_candidates: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    # Cross-encoder reranking (only when requested). Pinned commit; see README.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    rerank_max_length: int = Field(default=512, ge=16)
    rerank_candidates: int = Field(default=30, ge=1, le=100)
    # Grounded answers (Phase 5). The model is pinned by digest when llm_model_digest is
    # set; Ollama tags are mutable.
    llm_provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.2:3b"
    llm_model_digest: str | None = None
    llm_context_window: int = Field(default=4096, ge=512)
    llm_max_answer_tokens: int = Field(default=512, ge=16)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_seed: int = 7
    llm_timeout_seconds: float = Field(default=300.0, gt=0)
    answer_top_k: int = Field(default=12, ge=1, le=50)
    answer_budget_tokens: int = Field(default=1500, ge=100)
    answer_max_sources: int = Field(default=8, ge=1, le=20)
    answer_max_per_document: int = Field(default=3, ge=1)
    answer_rerank: bool = True
    # Publication authority policy; see rag.answering.authority.AuthorityPolicy.
    authority_policy: str = "current=include,unknown=include,superseded=demote,withdrawn=demote"
    # Retrieval-only sufficiency thresholds, calibrated on the TUNING set only (38 retrieval
    # questions + 10 unanswerable tuning questions; reports 20261027T213425Z and
    # 20261027T213448Z). With reranking the top logit decides (balanced accuracy 0.987);
    # without it, term coverage decides (0.755). See evals/datasets/nist-answers/README.md.
    sufficiency_min_coverage: float = Field(default=0.7, ge=0, le=1)
    sufficiency_min_rerank: float | None = -2.0
    answer_store_question: bool = True

    @model_validator(mode="after")
    def _check_lease(self) -> Self:
        if self.lease_seconds < 3 * self.poll_seconds:
            raise ValueError("lease_seconds must be at least three times poll_seconds")
        return self
