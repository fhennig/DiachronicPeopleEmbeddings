#! /usr/bin/env python3
import calendar
import logging
import re
import json
import time
import os.path
import gzip
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Tuple


API_URL = "https://content.guardianapis.com"
SEARCH = 'search'


logger = logging.getLogger(__name__)


def get_range_for_month(year: int, month: int) -> Tuple[str, str]:
    """Takes a year and a month and returns start and end of the month as
    string dates."""
    _, days = calendar.monthrange(year, month)
    start = f"{year}-{month:02}-01"
    end = f"{year}-{month:02}-{days:02}"
    return (start, end)


def get_date_ranges(start_year: int, start_month: int,
                    end_year: int, end_month: int) -> List[Tuple[str, str]]:
    """Returns a list of tuples of dates for the given time period.
    Start and end are inclusive, returns a tuple for each month."""
    assert start_year <= end_year
    results = []
    if start_year == end_year:
        for month in range(start_month, end_month + 1):
            results.append(get_range_for_month(start_year, month))
    else:
        for month in range(start_month, 12 + 1):
            results.append(get_range_for_month(start_year, month))
        for year in range(start_year + 1, end_year):
            for month in range(1, 12 + 1):
                results.append(get_range_for_month(year, month))
        for month in range(1, end_month + 1):
            results.append(get_range_for_month(end_year, month))
    return results


def test_get_date_ranges():
    months = get_date_ranges(2000, 1, 2000, 1)
    assert len(months) == 1
    months = get_date_ranges(2000, 1, 2000, 12)
    assert len(months) == 12
    months = get_date_ranges(2000, 1, 2001, 12)
    assert len(months) == 24
    months = get_date_ranges(2000, 1, 2019, 12)
    assert len(months) == 240


class GuardianApiCrawler:
    def __init__(self):
        self._api_key = os.getenv("API_KEY")
        if self._api_key is None:
            raise Exception("API_KEY is not given!")

    def _full_response_params(self):
        return {
            'api-key': self._api_key,
            'format': 'json',
            'page-size': 50,
            'page': 1,
            'use-date': 'published',
            'show-fields': 'all',
            'show-tags': 'all',
            'show-section': 'true',
            'show-blocks': 'all',
            'show-elements': 'all',
            'show-references': 'all',
            'show-rights': 'all'
        }

    def _get_api_results(self, start_date: str, end_date: str):
        """Returns an iterator of results for the given time period."""
        url = f"{API_URL}/{SEARCH}"
        parameters = self._full_response_params()
        parameters['from-date'] = start_date
        parameters['to-date'] = end_date
        max_pages = 10000000
        with tqdm(total=max_pages) as pbar:
            while parameters['page'] <= max_pages:
                pbar.update(1)
                response = requests.get(url, params=parameters)
                if response.status_code != 200:
                    print(response.content)
                    raise Exception()
                j = json.loads(response.content)['response']
                max_pages = j['pages']
                pbar.total = max_pages
                parameters['page'] += 1
                # save results
                for result in j['results']:
                    yield result
                time.sleep(0.1)

    def download(self, start_year: int, start_month: int,
                 end_year: int, end_month: int, directory: str):
        months = get_date_ranges(start_year, start_month, end_year, end_month)
        for start, end in tqdm(months, desc="Months"):
            path = os.path.join(directory, f"{start[:7]}.json.gz")
            if os.path.exists(path):
                logger.info("Skipping %s, already exists", path)
                continue
            # get the results
            articles = list(self._get_api_results(start, end))
            with gzip.open(path, 'wb') as f:
                bs = json.dumps(articles).encode('UTF-8')
                f.write(bs)
