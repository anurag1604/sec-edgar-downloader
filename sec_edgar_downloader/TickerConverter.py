from functools import lru_cache
from typing import ClassVar, Dict, Optional

from sec_cik_mapper import MutualFundMapper, StockMapper


class TickerConverter:
    ticker_to_cik_mapping: ClassVar[Dict[str, str]]

    def __init__(self):
        # Delay obtaining the mapping until a conversion is required
        self.ticker_to_cik_mapping = None

    @lru_cache
    def _get_cik_cached(self, ticker: str) -> Optional[str]:
        # Initialize and store ticker to CIK mapping
        if self.ticker_to_cik_mapping is None:
            stock_mapper = StockMapper()
            mutual_fund_mapper = MutualFundMapper()
            self.ticker_to_cik_mapping = dict(
                stock_mapper.ticker_to_cik,
                **mutual_fund_mapper.ticker_to_cik,
            )

        return self.ticker_to_cik_mapping.get(ticker)

    def to_cik(self, ticker: str) -> str:
        cik = self._get_cik_cached(ticker)

        if cik is None:
            raise ValueError(
                f"Unable to convert the ticker {ticker!r} to a CIK. "
                "Please use the official SEC CIK lookup tool to find the "
                "CIK and retry the download: "
                "sec.gov/edgar/searchedgar/cik.htm"
            )

        return cik
