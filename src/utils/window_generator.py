import pandas as pd
from datetime import datetime, timedelta

def generate_windows(full_df, train_days=90, test_days=30, anchor_most_recent=True):
    """
    Generate non-overlapping training and testing windows.
    
    Args:
        full_df: DataFrame with DatetimeIndex containing historical data
        train_days: Number of days in training window
        test_days: Number of days in out-of-sample test window
        anchor_most_recent: If True, work backward from most recent data
                           If False, work forward from earliest data
    
    Returns:
        List of dicts with train_start, train_end, test_start, test_end timestamps
    """
    df = full_df.sort_index()
    start_date = df.index[0].to_pydatetime()
    end_date = df.index[-1].to_pydatetime()
    windows = []

    if anchor_most_recent:
        current_end = end_date
        while True:
            test_end = current_end
            test_start = test_end - timedelta(days=test_days)
            train_end = test_start - timedelta(days=1)
            train_start = train_end - timedelta(days=train_days)
            if train_start < start_date:
                break
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'window_id': len(windows) + 1
            })
            current_end = train_start - timedelta(days=1)
    else:
        current_start = start_date
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)
            if test_end > end_date:
                break
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'window_id': len(windows) + 1
            })
            current_start = test_end + timedelta(days=1)

    print(f"Generated {len(windows)} windows from {start_date.date()} to {end_date.date()}")
    for w in windows[:3]:
        print(f"Window {w['window_id']}: Train {w['train_start'].date()}–{w['train_end'].date()}, "
              f"Test {w['test_start'].date()}–{w['test_end'].date()}")
    if len(windows) > 3:
        print("...")
    return windows 