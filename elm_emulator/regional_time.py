"""Calendar helpers for regional replay and forcing files."""

from __future__ import annotations

_NOLEAP_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_in_month(year: int, month: int, calendar: str = "gregorian") -> int:
    """Return days in month for the requested replay calendar."""
    if calendar == "noleap":
        return _NOLEAP_MONTH_LENGTHS[int(month) - 1]
    import calendar as _cal
    return _cal.monthrange(int(year), int(month))[1]


def _day_of_year_start(year: int, month: int, calendar: str = "gregorian") -> int:
    return sum(_days_in_month(year, m, calendar) for m in range(1, int(month))) + 1


def _days_in_year(year: int, calendar: str = "gregorian") -> int:
    if calendar == "noleap":
        return 365
    import calendar as _cal
    return 366 if _cal.isleap(int(year)) else 365
