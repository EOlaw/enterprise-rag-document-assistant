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

    @model_validator(mode="after")
    def _check_lease(self) -> Self:
        if self.lease_seconds < 3 * self.poll_seconds:
            raise ValueError("lease_seconds must be at least three times poll_seconds")
        return self
