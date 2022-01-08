"""Provides a :class:`Downloader` class for downloading SEC EDGAR filings."""

from pathlib import Path
from typing import ClassVar, List, Optional, Union, Dict

from ._constants import DATE_FORMAT_TOKENS, DEFAULT_AFTER_DATE, DEFAULT_BEFORE_DATE
from ._constants import SUPPORTED_FORMS as _SUPPORTED_FORMS
from ._utils import (
    download_filings,
    get_filings_to_download,
    is_cik,
    validate_date_format,
    validate_forms,
)

from sec_cik_mapper import StockMapper, MutualFundMapper


class Downloader:
    """A :class:`Downloader` object.

    :param download_folder: relative or absolute path to download location.
        Defaults to the current working directory.

    Usage::

        >>> from sec_edgar_downloader import Downloader

        # Download to current working directory
        >>> dl = Downloader()

        # Download to relative or absolute path
        >>> dl = Downloader("/path/to/valid/save/location")
    """

    ticker_to_cik_mapping: ClassVar[Dict[str, str]] = None

    supported_forms: ClassVar[List[str]] = sorted(_SUPPORTED_FORMS)

    def __init__(self, download_folder: Union[str, Path, None] = None) -> None:
        """Constructor for the :class:`Downloader` class."""
        if download_folder is None:
            self.download_folder = Path.cwd()
        elif isinstance(download_folder, Path):
            self.download_folder = download_folder
        else:
            self.download_folder = Path(download_folder).expanduser().resolve()

    def _convert_ticker_to_cik(self, ticker: str) -> str:
        # Initialize and cache ticker_to_cik_mapping
        if self.ticker_to_cik_mapping is None:
            stock_mapper = StockMapper()
            mutual_fund_mapper = MutualFundMapper()
            self.ticker_to_cik_mapping = dict(
                stock_mapper.ticker_to_cik,
                **mutual_fund_mapper.ticker_to_cik,
            )

        cik = self.ticker_to_cik_mapping.get(ticker)

        if cik is None:
            raise ValueError(
                f"Unable to convert the ticker {ticker!r} to a CIK. "
                "Please use the official SEC CIK lookup tool to find the CIK and retry the download: "
                "sec.gov/edgar/searchedgar/cik.htm"
            )

        return cik

    def get(
        self,
        forms: Union[str, List[str]],
        ticker_or_cik: str,
        *,
        amount: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_amends: bool = True,
        download_details: bool = True,
    ) -> int:
        """Download filings and save them to disk.

        :param forms: filing types to download (e.g. 8-K, 10-K).
        :param ticker_or_cik: ticker or CIK to download filings for.
        :param amount: number of filings to download.
            Defaults to all available filings.
        :param start_date: start date of form YYYY-MM-DD after which to download filings.
            Defaults to 1994-01-01, the earliest date supported by EDGAR.
        :param end_date: end date of form YYYY-MM-DD before which to download filings.
            Defaults to today.
        :param include_amends: denotes whether or not to include filing amends (e.g. 8-K/A).
            Defaults to True.
        :param download_details: denotes whether or not to download human-readable and easily
            parseable filing detail documents (e.g. form 4 XML, 8-K HTML). Defaults to True.
        :return: number of filings downloaded.

        Usage::

            >>> from sec_edgar_downloader import Downloader
            >>> dl = Downloader()

            # Get all 8-K filings for Apple
            >>> dl.get("8-K", "AAPL")

            # Get all 8-K filings for Apple, including filing amends (8-K/A)
            >>> dl.get("8-K", "AAPL", include_amends=True)

            # Get all 8-K filings for Apple after January 1, 2017 and before March 25, 2017
            >>> dl.get("8-K", "AAPL", after="2017-01-01", before="2017-03-25")

            # Get the five most recent 10-K filings for Apple
            >>> dl.get("10-K", "AAPL", amount=5)

            # Get all 10-K filings for Apple, excluding the filing detail documents
            >>> dl.get("10-K", "AAPL", amount=1, download_details=False)

            # Get all 10-Q filings for Visa
            >>> dl.get("10-Q", "V")

            # Get all 13F-NT filings for the Vanguard Group
            >>> dl.get("13F-NT", "0000102909")

            # Get all 13F-HR filings for the Vanguard Group
            >>> dl.get("13F-HR", "0000102909")

            # Get all SC 13G filings for Apple
            >>> dl.get("SC 13G", "AAPL")

            # Get all SD filings for Apple
            >>> dl.get("SD", "AAPL")
        """
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
            cik = ticker_or_cik.zfill(10)
        else:
            cik = self._convert_ticker_to_cik(ticker_or_cik)

        if amount is not None:
            amount = max(int(amount), 1)

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

        if isinstance(forms, str):
            forms = [forms]

        validate_forms(forms)

        filings_to_download = get_filings_to_download(
            forms,
            cik,
            amount,
            start_date,
            end_date,
            include_amends,
        )

        num_unique_filings_downloaded = download_filings(
            self.download_folder,
            cik,
            filings_to_download,
            download_details,
        )

        return num_unique_filings_downloaded
