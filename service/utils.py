from datetime import datetime


def timestamp_to_date(timestamp_date: float) -> datetime:
    return datetime.fromtimestamp(timestamp_date)
