import calendar
from typing import Tuple, List
from tqdm import tqdm


def get_range_for_month(year: int, month: int) -> Tuple[str, str]:
    """Takes a year and a month and returns start and end of the month as
    string dates."""
    _, days = calendar.monthrange(year, month)
    start = f"{year}-{month:02}-01"
    end = f"{year}-{month:02}-{days:02}"
    return (start, end)


def get_slice_starts(start_year, start_month,
                     slice_count, slice_size) -> List[Tuple[int, int]]:
    res = []
    res.append((start_year, start_month))
    for i in range(1, slice_count):
        new_y, new_m = res[i-1]
        new_m += slice_size
        # WARNING!! Does not support slice_size bigger than 12.
        if new_m > 12:
            new_m -= 12
            new_y += 1
        res.append((new_y, new_m))
    return res


def get_years_months(start_year, start_month,
                     end_year, end_month, inclusive=True) -> List[Tuple[int, int]]:
    """Takes a start and end year+month and returns a list of those.
    Start and end included."""
    res = []
    cur_y, cur_m = start_year, start_month
    while cur_y != end_year or cur_m != end_month:
        res.append((cur_y, cur_m))
        cur_m += 1
        if cur_m == 13:
            cur_m = 1
            cur_y += 1
    if inclusive:
        res.append((end_year, end_month))
    return res


def get_years_months_dur(start_year, start_month, dur) -> List[Tuple[int, int]]:
    res = []
    res.append((start_year, start_month))
    for i in range(1, dur):
        n_y, n_m = res[i-1]
        n_m += 1
        if n_m == 13:
            n_m -= 12
            n_y += 1
        res.append((n_y, n_m))
    return res


def batch_list(l, batch_size):
    while l:
        yield l[:batch_size]
        l = l[batch_size:]


def get_years_month_blocks(start_year, start_month,
                           slice_count, slice_size) -> List[List[Tuple[int, int]]]:
    starts = get_slice_starts(start_year, start_month, slice_count, slice_size)
    res = list([get_years_months_dur(y, m, slice_size) for y, m in starts])
    return res

def write_tsv_file(rows, path):
    with open(path, 'w') as f:
        for row in tqdm(rows, leave=False, desc="Writing tsv file"):
            line = "\t".join([f"{cell}" for cell in row])
            f.write(f"{line}\n")


def list_of_dicts_to_dict(list_of_dicts):
    """Takes a list of dictionaries and merges them.
    This is useful for SQL responses."""
    res = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if k in res:
                raise Exception()
            res[k] = v
    return res
