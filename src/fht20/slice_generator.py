from enum import Enum
import calendar
import datetime
from typing import List, Tuple


MONTHLY = 1
QUARTERLY = 3
HALFYEARLY = 6
YEARLY = 12


def get_end(year, month, slice_size):
    month += slice_size
    if month > 12:
        month -= 12
        year += 1
    dt = datetime.datetime(year, month, 1)
    dt -= datetime.timedelta(days=1)
    return dt


def generate_slices(start_year: int, start_month: int,
                    slice_size: int, slice_count: int) -> List[Tuple[str, str]]:
    starts = [(start_year, start_month)]
    for i in range(slice_count - 1):
        cur_year, cur_month = starts[-1]
        new_year = cur_year
        new_month = cur_month + slice_size
        if new_month > 12:
            new_month -= 12
            new_year += 1
        starts.append((new_year, new_month))
    ranges = [(datetime.datetime(y, m, 1),
               get_end(y, m, slice_size))
              for y, m in starts]
    ranges = slices_as_strs(ranges)
    return ranges


def slices_as_strs(slices: List[Tuple[datetime.datetime, datetime.datetime]]) -> List[Tuple[str, str]]:
    format = "%Y-%m-%d"
    return [(s.strftime(format), e.strftime(format))
            for s, e in slices]
