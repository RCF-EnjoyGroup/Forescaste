"""Holiday feature engineer for Costa Rica and custom events."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_costa_rica_holidays(years: list[int] | None = None) -> pd.DatetimeIndex:
    """
    Get Costa Rica national holidays for specified years.

    Uses the `holidays` library if available, otherwise returns a static list.
    """
    if years is None:
        years = list(range(2020, 2030))

    try:
        import holidays
        cr_holidays = holidays.CR(years=years)
        return pd.DatetimeIndex(cr_holidays.keys())
    except (ImportError, AttributeError):
        # Fallback: static Costa Rica holidays
        logger.warning("holidays library not available. Using static Costa Rica calendar.")
        static_holidays = []
        for year in years:
            static_holidays.extend([
                f"{year}-01-01",  # New Year's Day
                f"{year}-04-11",  # Juan Santamaría Day
                f"{year}-05-01",  # Labor Day
                f"{year}-07-25",  # Guanacaste Day
                f"{year}-08-02",  # Virgin of the Angels Day
                f"{year}-08-15",  # Mother's Day
                f"{year}-09-15",  # Independence Day
                f"{year}-12-25",  # Christmas Day
            ])
        return pd.DatetimeIndex(pd.to_datetime(static_holidays))


class HolidayFeatureEngineer:
    """
    Create holiday and event indicator features.

    Parameters
    ----------
    pre_days : int
        Days before holiday to flag as "pre-holiday".
    post_days : int
        Days after holiday to flag as "post-holiday".
    custom_events : dict, optional
        Custom event dates: {name: [dates]}.
    """

    def __init__(
        self,
        pre_days: int = 3,
        post_days: int = 3,
        custom_events: dict[str, list[str]] | None = None,
    ):
        self.pre_days = pre_days
        self.post_days = post_days
        self.custom_events = custom_events or {}
        self._holiday_dates: pd.DatetimeIndex | None = None

    def fit(self, df: pd.DataFrame) -> "HolidayFeatureEngineer":
        """Learn date range and fetch holidays."""
        date_col = "date" if "date" in df.columns else df.index.name
        if "date" in df.columns:
            dates = df["date"]
        else:
            dates = df.index

        years = sorted(set(dates.dt.year.dropna()))
        self._holiday_dates = get_costa_rica_holidays(years)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday features."""
        df = df.copy()

        if "date" in df.columns:
            dt = df["date"]
        else:
            dt = pd.Series(df.index, index=df.index)

        if self._holiday_dates is None:
            self.fit(df)

        # Main holiday flag
        df["is_holiday"] = dt.isin(self._holiday_dates).astype(int)

        # Pre/post holiday windows
        df["is_pre_holiday"] = 0
        df["is_post_holiday"] = 0
        for hdate in self._holiday_dates:
            pre_mask = (dt >= hdate - pd.Timedelta(days=self.pre_days)) & (dt < hdate)
            post_mask = (dt > hdate) & (dt <= hdate + pd.Timedelta(days=self.post_days))
            df.loc[pre_mask, "is_pre_holiday"] = 1
            df.loc[post_mask, "is_post_holiday"] = 1

        # Days to next holiday
        df["days_to_holiday"] = self._compute_days_to_event(dt, self._holiday_dates)

        # Custom events
        for event_name, event_dates in self.custom_events.items():
            event_dt = pd.to_datetime(event_dates)
            df[f"is_{event_name}"] = dt.isin(event_dt).astype(int)

        logger.info("Holiday features added: is_holiday, is_pre_holiday, is_post_holiday, days_to_holiday")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def _compute_days_to_event(dt: pd.Series, event_dates: pd.DatetimeIndex) -> pd.Series:
        """Compute days until next event for each date."""
        result = pd.Series(np.nan, index=dt.index)
        for idx, d in dt.items():
            future = event_dates[event_dates >= d]
            if len(future) > 0:
                result[idx] = (future[0] - d).days
            else:
                result[idx] = 365  # Default if no future events
        return result


# Need numpy for _compute_days_to_event
import numpy as np  # noqa: E402
