import numpy as np
cimport numpy as np
import csv
import requests


cdef bint is_float(str string) noexcept:
    try:
        float(string)
        return True
    except ValueError:
        return False


cdef string_to_csv_and_numpy(
    str input_string, str delimiter=",", bint header=True
) noexcept:
    reader = csv.reader(input_string.splitlines(), delimiter=delimiter)

    # read the header separately
    headers = next(reader)
    columns = [[] for _ in headers]

    # read all rows of data
    for row in reader:
        for i, value in enumerate(row):
            columns[i].append(value)

    if not header:
        for i, value in enumerate(headers):
            columns[i].insert(0, value)

    # convert the collected columns to numpy arrays
    columns = [np.array(col, dtype=np.float64 if is_float(col[0]) else str)
        for col in columns]

    if header:
        return [np.array(headers, dtype=str), *columns]
    
    return columns


# TODO: add skiprows
cpdef list[np.ndarray] fetch_csv_to_numpy(
    str csv_url, list[int] columns=list(),
    str delimiter=",", bint header=True
) except *:
    # check if the URL is for a csv extension file
    if not csv_url.endswith('.csv'):
        raise ValueError("The file name does not have the .csv extension")

    # search file by URL
    try:
        response = requests.get(csv_url)
        if response.status_code != 200:
            raise requests.exceptions.RequestException(
                "Request with status code other than 200"
                f" ({response.status_code})"
            )
    except requests.exceptions.RequestException as error:
        print(f"Failed to fetch to url: {csv_url}.\n{error}")

    arrays = string_to_csv_and_numpy(response.text, delimiter, header)

    # fix the column list so that it works with headers
    if header:
        headers = [col for i, col in enumerate(arrays[0]) if i in columns]
        columns = [col + 1 for col in columns]
    else:
        headers = []

    # return only requested columns
    if not columns:
        return arrays
    else:
        data = [column for i, column in enumerate(arrays) if i in columns]
        return [headers, *data]

