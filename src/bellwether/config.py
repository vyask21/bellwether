"""Runtime configuration, loaded from environment / `.env`."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"


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
    anthropic_api_key: str = Field(default="", description="Used by the explanation layer only.")

    duckdb_path: Path = Field(default=DATA_DIR / "bellwether.duckdb")

    def require_eia_key(self) -> str:
        if not self.eia_api_key:
            raise RuntimeError(
                "EIA_API_KEY is not set. Register free at "
                "https://www.eia.gov/opendata/register.php then add it to .env"
            )
        return self.eia_api_key


settings = Settings()
