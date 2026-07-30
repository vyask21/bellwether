"""Source attribution for EIA content.

Two EIA policies govern this:

* API Terms of Service: identify EIA as the source, do not imply endorsement.
* Copyrights and Reuse: acknowledgments should carry a date, in the style
  "Source: U.S. Energy Information Administration (Oct 2008)".

EIA content is public domain and freely redistributable, so the obligation is
acknowledgment rather than permission. Keeping the strings here means every display
surface emits the same compliant text.

See docs/EIA_COMPLIANCE.md.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

AGENCY = "U.S. Energy Information Administration"

# Neutral phrasing only. "Courtesy of", "powered by", and "in partnership with" can read as
# endorsement, which the Terms of Service prohibit.
EIA_SOURCE = f"Source: {AGENCY} (EIA), https://www.eia.gov/opendata/"

# Attached to anything derived from EIA data. Forecasts are model output, not EIA content,
# and the TOS forbids representing modified content as EIA's.
DERIVED_DISCLAIMER = (
    "Forecasts and derived values are produced by this project, not by EIA. "
    "They are not authoritative and carry no warranty."
)

NOT_AFFILIATED = "This project is not affiliated with or endorsed by EIA."


def eia_acknowledgment(retrieved: date | datetime | None = None) -> str:
    """Dated acknowledgment in EIA's documented style.

    EIA's reuse policy asks for a date. APIv2 does not expose one: the FAQ states the
    update field was removed for performance and that data updates constantly rather than
    on a schedule. So the date given is when we retrieved the data, and it is labelled
    "retrieved" rather than presented as a publication date we do not have.
    """
    moment = retrieved or datetime.now(UTC)
    if isinstance(moment, datetime):
        moment = moment.astimezone(UTC).date() if moment.tzinfo else moment.date()
    return f"Source: {AGENCY} (retrieved {moment:%b %Y})"


def attribution_block(retrieved: date | datetime | None = None) -> str:
    """Full notice for any surface showing observations alongside derived values."""
    return "\n".join(
        [
            eia_acknowledgment(retrieved),
            NOT_AFFILIATED,
            DERIVED_DISCLAIMER,
        ]
    )
