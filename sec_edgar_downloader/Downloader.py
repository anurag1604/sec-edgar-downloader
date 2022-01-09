"""Provides a :class:`Downloader` class for downloading SEC EDGAR filings."""

from pathlib import Path
from typing import ClassVar, List, Optional, Union

import pandas as pd

from ._constants import SUPPORTED_FORMS as _SUPPORTED_FORMS
from ._utils import download_filings, get_download_urls, get_filings_to_download
from ._validation import validate_dates, validate_forms, validate_ticker_or_cik


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

    supported_forms: ClassVar[List[str]] = sorted(_SUPPORTED_FORMS)

    def __init__(self, download_folder: Union[str, Path, None] = None) -> None:
        """Constructor for the :class:`Downloader` class."""
        if download_folder is None:
            self.download_folder = Path.cwd()
        elif isinstance(download_folder, Path):
            self.download_folder = download_folder
        else:
            self.download_folder = Path(download_folder).expanduser().resolve()

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
        only_dataframe: bool = False,
    ) -> pd.DataFrame:
        """Download filings and save them to disk.

        :param forms: filing types to download (e.g. 8-K, 10-K).
        :param ticker_or_cik: ticker or CIK to download filings for.
        :param amount: number of filings to download.
            Defaults to all available filings.
        :param start_date: start date of form YYYY-MM-DD after which to download filings.
            Defaults to 1994-01-01, the earliest date supported by EDGAR.
        :param end_date: end date of form YYYY-MM-DD before which to download filings.
            Defaults to today.
        :param include_amends: whether to include filing amends (e.g. 8-K/A).
            Defaults to True.
        :param download_details: whether to download human-readable and easily
            parseable filing detail documents (e.g. form 4 XML, 8-K HTML). Defaults to True.
        :param only_dataframe: return a dataframe containing data that matches the provided
            parameters without downloading the filings to disk. Defaults to False.
            By default, filings will be downloaded and the dataframe will be returned.

        :return: pandas dataframe with the columns form, accessionNumber, filingDate,
            fullSubmissionUrl, and filingDetailsUrl.

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
        cik = validate_ticker_or_cik(ticker_or_cik)

        # If amount is not specified, download all available filings
        # for the specified input parameters.
        # Else, we need to ensure that the specified amount is valid.
        if amount is not None:
            amount = max(int(amount), 1)

        # Validate start and end dates and set defaults if None
        start_date, end_date = validate_dates(start_date, end_date)

        if isinstance(forms, str):
            forms = [forms]

        validate_forms(forms)

        filings_df = get_filings_to_download(
            forms,
            cik,
            amount,
            start_date,
            end_date,
            include_amends,
        )

        download_urls_df = get_download_urls(cik, filings_df)

        if not only_dataframe:
            download_filings(
                self.download_folder,
                cik,
                download_urls_df,
                download_details,
            )

        return download_urls_df
