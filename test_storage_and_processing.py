#!/usr/bin/env python

import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import unittest
from coding_exercise.storage_and_processing import StorageAndProcessing
"""
Test Cases for Storage and Processing

Repository:
coding_exercise

History:
- 12/30/2021 - CSM - original creation
"""


class TestStorageAndProcessing(unittest.TestCase):

    def test_add_input_integer(self):
        s_and_p = StorageAndProcessing()
        s_and_p.add_input(1)

    def test_add_input_list(self):
        s_and_p = StorageAndProcessing()
        s_and_p.add_input([1, 2, 3])

    def test_add_list_of_list(self):
        now = datetime.now()
        np_now = np.datetime64(now)
        input_list = [[np_now, 1], [np_now, 2]]
        s_and_p = StorageAndProcessing()
        s_and_p.add_input(input_list)

    def test_add_pandas_df(self):
        now = datetime.now()
        datetimes = pd.date_range(now - timedelta(days=7), now, freq='S')
        data = np.random.randint(0, high=10000000, size=len(datetimes))
        df = pd.DataFrame({'datetime': datetimes, 'integer': data})
        s_and_p = StorageAndProcessing()
        s_and_p.add_input(df)

    def test_add_pandas_df(self):
        now = datetime.now()
        datetimes = pd.date_range(now - timedelta(days=7), now, freq='S')
        data = np.random.randint(0, high=10000000, size=len(datetimes))
        df = pd.DataFrame({'datetime': datetimes, 'integer': data})
        df.set_index('datetime')
        s_and_p = StorageAndProcessing()
        s_and_p.add_input(df)

    def test_mean_std_dev(self):
        s_and_p = StorageAndProcessing()
        s_and_p.add_input([1, 2, 3])
        results = s_and_p.get_mean_and_std_deviation(1)
        mean = results.get('mean')
        std_dev = results.get('std_dev')
        assert mean == 2 and std_dev == 0.8164965809277263

    def test_add_giant_mean_and_std_dev(self):
        now = datetime.now()
        datetimes = pd.date_range(now - timedelta(days=30), now, freq='S')
        data = np.random.randint(0, high=1000000, size=len(datetimes))
        df = pd.DataFrame({'datetime': datetimes, 'integer': data})
        df.set_index('datetime')
        s_and_p = StorageAndProcessing()
        s_and_p.add_input(df)
        input_seconds = timedelta(days=30).total_seconds()
        m_seconds = random.random()
        m_seconds += random.randint(0, int(input_seconds))
        print(m_seconds)
        results = s_and_p.get_mean_and_std_deviation(m_seconds, test=True)
        print(results)


if __name__ == "__main__":
    unittest.main()
