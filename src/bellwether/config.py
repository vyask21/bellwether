"""Runtime configuration, loaded from environment / `.env`."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# The committed Parquet mirror of the DuckDB source tables. Distinct from SNAPSHOT_DIR,
# which is a gitignored scratch export, and from `snapshot/`, which is the narrow
# dashboard cache. This is the one directory that survives a machine: a scheduled run on
# an ephemeral runner rebuilds the store from it, tops it up, and writes it back.
STORE_DIR = PROJECT_ROOT / "store"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    eia_api_key: str = Field(default="", description="EIA v2 API key")
    nws_user_agent: str = Field(
        default="(bellwether, contact-unset@example.com)",
        description="NWS requires a User-Agent identifying the app and a contact address.",
    )
    llm_api_key: str = Field(default="", description="Used by the optional model brief only.")

    duckdb_path: Path = Field(default=DATA_DIR / "bellwether.duckdb")

    def require_eia_key(self) -> str:
        if not self.eia_api_key:
            raise RuntimeError(
                "EIA_API_KEY is not set. Register free at "
                "https://www.eia.gov/opendata/register.php then add it to .env"
            )
        return self.eia_api_key


settings = Settings()
