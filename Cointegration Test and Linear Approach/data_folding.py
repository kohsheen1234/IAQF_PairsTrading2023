from datetime import datetime
from typing import Sequence

import pandas as pd
from attrs import define


@define
class Fold:
    formation_set: Sequence[datetime] = None
    test_set: Sequence[datetime] = None


class DataFolder:
    def __init__(self, timestamps: Sequence[datetime], formation_period: int, test_period: int):
        self._formation_period = formation_period
        self._test_period = test_period
        self._timestamps = timestamps

    def __iter__(self):
        current_idx = self._formation_period
        max_timestamps = len(self._timestamps)
        while current_idx <= max_timestamps:

            formation_start = current_idx - self._formation_period
            formation_end = test_start = current_idx
            test_end = formation_end + self._test_period

            current_idx += self._test_period
            yield Fold(formation_set=self._timestamps[formation_start: formation_end],
                       test_set=self._timestamps[test_start: test_end])

    def __len__(self):
        return int((len(self._timestamps) - self._formation_period) / self._test_period) + 1


if __name__ == "__main__":
    spread = pd.read_csv("spread.csv")
    ts = spread["Date"].values
    folder = DataFolder(timestamps=ts, formation_period=504, test_period=30)
    idx = 0
    for fold in folder:
        idx += 1
        print(f"Train set:  start={fold.formation_set[0]} \t end={fold.formation_set[-1]} \n"
              f"Test set: start={fold.test_set[0]} \t end={fold.test_set[-1]}.")
    assert idx == len(folder)
