"""Input validation functions."""
from datetime import datetime
from typing import List, Optional, Tuple

from ._constants import (
    DATE_FORMAT_TOKENS,
    DEFAULT_AFTER_DATE,
    DEFAULT_BEFORE_DATE,
    SUPPORTED_FORMS,
)
from .TickerConverter import TickerConverter

ticker_converter = TickerConverter()


def is_cik(ticker_or_cik: str) -> bool:
    try:
        int(ticker_or_cik)
        return True
    except ValueError:
        return False


def validate_date_format(date_format: str) -> None:
    error_msg_base = "Please enter a date string of the form YYYY-MM-DD."

    if not isinstance(date_format, str):
        raise TypeError(error_msg_base)

    try:
        datetime.strptime(date_format, DATE_FORMAT_TOKENS)
    except ValueError as exc:
        # Re-raise with custom error message
        raise ValueError(f"Incorrect date format. {error_msg_base}") from exc


def validate_forms(forms: List[str]) -> None:
    unsupported_forms = set(forms) - SUPPORTED_FORMS
    if unsupported_forms:
        filing_options = ", ".join(sorted(SUPPORTED_FORMS))
        raise ValueError(
            f"{','.join(unsupported_forms)!r} filings are not supported. "
            f"Please choose from the following: {filing_options}."
        )


def validate_dates(
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[str, str]:
    # SEC allows for filing searches from 1994 onwards
    if start_date is None:
        start_date = DEFAULT_AFTER_DATE.strftime(DATE_FORMAT_TOKENS)
    else:
        validate_date_format(start_date)

        if start_date < DEFAULT_AFTER_DATE.strftime(DATE_FORMAT_TOKENS):
            raise ValueError(
                f"Filings cannot be downloaded prior to {DEFAULT_AFTER_DATE.year}. "
                f"Please enter a date on or after {DEFAULT_AFTER_DATE}."
            )

    if end_date is None:
        end_date = DEFAULT_BEFORE_DATE.strftime(DATE_FORMAT_TOKENS)
    else:
        validate_date_format(end_date)

    if start_date > end_date:
        raise ValueError(
            "Invalid after and before date combination. "
            "Please enter an after date that is less than the before date."
        )

    return start_date, end_date


def validate_ticker_or_cik(ticker_or_cik: str):
    ticker_or_cik = str(ticker_or_cik).strip().upper()

    # Check for blank tickers or CIKs
    if not ticker_or_cik:
        raise ValueError("Invalid ticker or CIK. Please enter a non-blank value.")

    # Detect CIKs and ensure that they are properly zero-padded
    if is_cik(ticker_or_cik):
        if len(ticker_or_cik) > 10:
            raise ValueError("Invalid CIK. CIKs must be at most 10 digits long.")
        # Pad CIK with 0s to ensure that it is exactly 10 digits long
        # The SEC EDGAR API requires zero-padded CIKs
        return ticker_or_cik.zfill(10)

    return ticker_converter.to_cik(ticker_or_cik)
