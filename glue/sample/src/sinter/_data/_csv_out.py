import collections
import csv
import io
import json
from typing import Any, Optional


def escape_csv(text: Any, width: Optional[int]) -> str:
    output = io.StringIO()
    csv.writer(output).writerow([text])
    text = output.getvalue().strip()
    if width is not None:
        text = text.rjust(width)
    return text


def csv_line(*,
             shots: Any,
             errors: Any,
             discards: Any,
             seconds: Any,
             decoder: Any,
             strong_id: Any,
             json_metadata: Any,
             custom_counts: Any,
             is_header: bool = False) -> str:
    if isinstance(seconds, float):
        if seconds < 1:
            seconds = f'{seconds:0.3f}'
        elif seconds < 10:
            seconds = f'{seconds:0.2f}'
        else:
            seconds = f'{seconds:0.1f}'
    if not is_header:
        json_metadata = json.dumps(json_metadata,
                                   separators=(',', ':'),
                                   sort_keys=True)
        if custom_counts:
            # Ensure all custom_counts values are integers before dumping to JSON
            for k in custom_counts:
                custom_counts[k] = int(custom_counts[k])
            custom_counts = escape_csv(
                json.dumps(custom_counts,
                           separators=(',', ':'),
                           sort_keys=True), None)
        else:
            custom_counts = ''


    shots = escape_csv(shots, 10)
    if isinstance(errors, (dict, collections.Counter)):
        errors = json.dumps(errors, separators=(',', ':'), sort_keys=True)
    errors = escape_csv(errors, 10)
    discards = escape_csv(discards, 10)
    seconds = escape_csv(seconds, 8)
    decoder = escape_csv(decoder, None)
    strong_id = escape_csv(strong_id, None)
    json_metadata = escape_csv(json_metadata, None)
    return (f'{shots},'
            f'{errors},'
            f'{discards},'
            f'{seconds},'
            f'{decoder},'
            f'{strong_id},'
            f'{json_metadata},'
            f'{custom_counts}')


CSV_HEADER = csv_line(shots='shots',
                      errors='errors',
                      discards='discards',
                      seconds='seconds',
                      strong_id='strong_id',
                      decoder='decoder',
                      json_metadata='json_metadata',
                      custom_counts='custom_counts',
                      is_header=True)
