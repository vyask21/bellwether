"""Source attribution for third-party content.

Two agencies publish into this project, under different policies, and their notices are
kept apart so an EIA acknowledgment never lands on a NOAA value or the reverse.

Two EIA policies govern the EIA half:

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

# NOAA. Its data is a US Government work and so public domain, which makes the obligation
# citation rather than permission, as with EIA. NCEI asks that the dataset be named, not
# just the agency: "NOAA" alone does not identify which of its archives a number came from.
NOAA_AGENCY = "NOAA National Centers for Environmental Information"
NOAA_DATASET = "Integrated Surface Database (ISD)"
NOAA_SOURCE = (
    f"Source: {NOAA_AGENCY}, {NOAA_DATASET}, "
    "https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database"
)
NOAA_NOT_AFFILIATED = "This project is not affiliated with or endorsed by NOAA."


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


def noaa_acknowledgment(retrieved: date | datetime | None = None) -> str:
    """Dated acknowledgment for NCEI content, matching the EIA one in shape.

    The date is when we retrieved the archive, and it carries more weight here than for
    EIA: NCEI revises the record as late reports arrive and quality control runs, so two
    retrievals of the same hour can differ.
    """
    moment = retrieved or datetime.now(UTC)
    if isinstance(moment, datetime):
        moment = moment.astimezone(UTC).date() if moment.tzinfo else moment.date()
    return f"Source: {NOAA_AGENCY}, {NOAA_DATASET} (retrieved {moment:%b %Y})"


def attribution_block(retrieved: date | datetime | None = None) -> str:
    """Full notice for any surface showing observations alongside derived values."""
    return "\n".join(
        [
            eia_acknowledgment(retrieved),
            NOT_AFFILIATED,
            noaa_acknowledgment(retrieved),
            NOAA_NOT_AFFILIATED,
            DERIVED_DISCLAIMER,
        ]
    )
