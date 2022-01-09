"""Utility functions for the downloader class."""
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from faker import Faker
from pyrate_limiter import Duration, Limiter, RequestRate
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ._constants import (
    FILING_DETAILS_FILENAME_STEM,
    FILING_FULL_SUBMISSION_FILENAME,
    MAX_REQUESTS_PER_SECOND,
    MAX_RETRIES,
    ROOT_SAVE_FOLDER_NAME,
    SEC_EDGAR_ARCHIVES_BASE_URL,
    SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,
    SEC_EDGAR_SUBMISSIONS_API_BASE_URL,
)

# Rate limiter
rate = RequestRate(MAX_REQUESTS_PER_SECOND, Duration.SECOND)
limiter = Limiter(rate)

# Object for generating fake user-agent strings
fake = Faker()

# Specify max number of request retries
# https://stackoverflow.com/a/35504626/3820660
retries = Retry(
    total=MAX_RETRIES,
    backoff_factor=SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,
    status_forcelist=[403, 500, 502, 503, 504],
)


@limiter.ratelimit(delay=True)
def rate_limited_get_request(client: requests.Session, url: str):
    resp = client.get(url)
    resp.raise_for_status()
    return resp


def get_submissions_for_cik(query_cik: str) -> pd.DataFrame:
    client = requests.Session()
    client.headers.update(
        {
            "User-Agent": generate_random_user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
    )
    client.mount("http://", HTTPAdapter(max_retries=retries))
    client.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        resp = rate_limited_get_request(
            client, f"{SEC_EDGAR_SUBMISSIONS_API_BASE_URL}/CIK{query_cik}.json"
        ).json()
        # Every company will have recent filings
        submissions = [pd.DataFrame(resp["filings"]["recent"])]

        # Companies with >1000 filings will have an extra list of filings defined,
        # so we need to iterate through them and concatenate them to the origin dataframe
        extra_filings = resp["filings"]["files"]
        for filing in extra_filings:
            sec_resource = filing["name"]
            resp = rate_limited_get_request(
                client, f"{SEC_EDGAR_SUBMISSIONS_API_BASE_URL}/{sec_resource}"
            ).json()
            submissions.append(pd.DataFrame(resp))

        return (
            pd.concat(submissions, ignore_index=True)
            if len(submissions) > 1
            else submissions[0]
        )
    finally:
        client.close()


def get_filings_to_download(
    forms: List[str],
    cik: str,
    amount: int,
    start_date: str,
    end_date: str,
    include_amends: bool,
) -> pd.DataFrame:
    submissions = get_submissions_for_cik(cik)
    filtered_submissions = filter_dataframe(
        submissions, forms, amount, start_date, end_date, include_amends
    )
    return filtered_submissions


def filter_dataframe(
    submissions: pd.DataFrame,
    forms: Union[str, List[str]],
    amount: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
    include_amends: bool,
) -> pd.DataFrame:
    submissions.filingDate = pd.to_datetime(submissions.filingDate)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    mask = submissions.form.isin(forms) & submissions.filingDate.isin(date_range)
    if not include_amends:
        # Exclude filing amends
        mask &= ~submissions.primaryDocDescription.str.endswith("/A")

    if amount is None:
        # Download all submissions after applying mask
        filtered_submissions = submissions[mask]
    else:
        # Download the first n number of forms after applying mask
        # For example: Select forms 6-K, 8-K, and 10-K and specify the
        # amount to be 3. If the submissions with mask contains 15
        # filings (5 of each form type), applying the following query
        # will download a total of 9 filings, 3 of each form type.
        filtered_submissions = submissions[mask].groupby("form").head(amount)

    return filtered_submissions.reset_index()


def get_download_urls(cik: str, filings_to_download: pd.DataFrame) -> pd.DataFrame:
    download_urls = defaultdict(list)
    for filing in filings_to_download.itertuples():
        download_urls["accessionNumber"].append(filing.accessionNumber)
        download_urls["form"].append(filing.form)
        download_urls["filingDate"].append(filing.filingDate)

        accession_number_no_dashes = filing.accessionNumber.replace("-", "", 2)
        submission_base_url = (
            f"{SEC_EDGAR_ARCHIVES_BASE_URL}/{cik}/{accession_number_no_dashes}"
        )

        download_urls["fullSubmissionUrl"].append(
            f"{submission_base_url}/{filing.accessionNumber}.txt"
        )

        if filing.primaryDocument:
            primary_doc = Path(filing.primaryDocument)
            download_urls["filingDetailsUrl"].append(
                f"{submission_base_url}/{primary_doc.name}"
            )
        else:
            download_urls["filingDetailsUrl"].append("")

    return pd.DataFrame(download_urls)


def download_filings(
    download_folder: Path,
    cik: str,
    download_urls_df: pd.DataFrame,
    include_filing_details: bool,
):
    client = requests.Session()
    client.headers.update(
        {
            "User-Agent": generate_random_user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
    )
    client.mount("http://", HTTPAdapter(max_retries=retries))
    client.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        for filing in download_urls_df.itertuples():
            try:
                download_and_save_filing(
                    client,
                    download_folder,
                    cik,
                    filing.accessionNumber,
                    filing.form,
                    filing.fullSubmissionUrl,
                    FILING_FULL_SUBMISSION_FILENAME,
                )
            except requests.exceptions.HTTPError as e:  # pragma: no cover
                print(
                    "Skipping full submission download for "
                    f"{filing.accessionNumber!r} due to network error: {e}."
                )

            if include_filing_details and filing.filingDetailsUrl:
                # Get filename from SEC URL in order to extract the appropriate extension
                url_path = str(urlparse(filing.filingDetailsUrl).path)
                sec_filename = url_path.rstrip("/").rsplit("/", 1).pop()
                extension = Path(sec_filename).suffix.replace("htm", "html")
                save_filename = f"{FILING_DETAILS_FILENAME_STEM}{extension}"
                try:
                    download_and_save_filing(
                        client,
                        download_folder,
                        cik,
                        filing.accessionNumber,
                        filing.form,
                        filing.filingDetailsUrl,
                        save_filename,
                        resolve_urls=True,
                    )
                except requests.exceptions.HTTPError as e:  # pragma: no cover
                    print(
                        f"Skipping filing detail download for "
                        f"{filing.accessionNumber!r} due to network error: {e}."
                    )
    finally:
        client.close()


def resolve_relative_urls_in_filing(
    filing_text: bytes, download_url: str
) -> Union[str, BeautifulSoup]:
    soup = BeautifulSoup(filing_text, "lxml")
    base_url = f"{download_url.rsplit('/', 1)[0]}/"

    for url in soup.find_all("a", href=True):
        # Do not resolve a URL if it is a fragment or it already contains a full URL
        if url["href"].startswith("#") or url["href"].startswith("http"):
            continue
        url["href"] = urljoin(base_url, url["href"])

    for image in soup.find_all("img", src=True):
        image["src"] = urljoin(base_url, image["src"])

    if soup.original_encoding is None:  # pragma: no cover
        return soup

    return soup.encode(soup.original_encoding)


def download_and_save_filing(
    client: requests.Session,
    download_folder: Path,
    ticker_or_cik: str,
    accession_number: str,
    filing_type: str,
    download_url: str,
    save_filename: str,
    *,
    resolve_urls: bool = False,
) -> None:
    filing_text = rate_limited_get_request(client, download_url).content

    # Only resolve URLs in HTML files
    if resolve_urls and Path(save_filename).suffix == ".html":
        filing_text = resolve_relative_urls_in_filing(filing_text, download_url)

    # Create all parent directories as needed and write content to file
    save_path = (
        download_folder
        / ROOT_SAVE_FOLDER_NAME
        / ticker_or_cik
        / filing_type
        / accession_number
        / save_filename
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(filing_text)


def generate_random_user_agent() -> str:
    return f"{fake.first_name()} {fake.last_name()} {fake.email()}"
