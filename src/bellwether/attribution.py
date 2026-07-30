"""Source attribution for EIA content.

The EIA API Terms of Service require identifying EIA as the source of API content, and
forbid wording that implies EIA endorsement. Keeping the strings in one module means every
display surface emits the same compliant text.

See docs/EIA_COMPLIANCE.md.
"""

from __future__ import annotations

# Neutral phrasing only. "Courtesy of", "powered by", and "in partnership with" can read as
# endorsement, which the TOS prohibits.
EIA_SOURCE = "Source: U.S. Energy Information Administration (EIA), https://www.eia.gov/opendata/"

# Attached to anything derived from EIA data. Forecasts are model output, not EIA content,
# and the TOS forbids representing modified content as EIA's.
DERIVED_DISCLAIMER = (
    "Forecasts and derived values are produced by this project, not by EIA. "
    "They are not authoritative and carry no warranty."
)


def attribution_block() -> str:
    """Both notices, for any surface that shows observations alongside forecasts."""
    return f"{EIA_SOURCE}\n{DERIVED_DISCLAIMER}"
