#!/usr/bin/env python

"""
Stores and Processes Very Large Dataframes that would exceed user memory usage and processing power

Repository:
coding_exercise

History:
- 12/29/2021 - CSM - original creation
"""

import random
import os
import sys
import json
import shutil
from tempfile import mkdtemp
import numpy as np
import pandas as pd
import vaex as vx
import dask.dataframe as dk
from collections import OrderedDict
from datetime import datetime, timedelta


class StorageAndProcessing:
    """
    Stores and Processes Very Large Dataframes
    """
    COLUMNS = ['datetime', 'integer']
    INDEX = 'datetime'
    EXPRESSION = 'integer'
    STORAGE_FOLDER = mkdtemp()
    ALLOWED_VALUES = {
        'processing_type': ['vaex'],  # might use 'dask' in future, 'pandas' if data small enough
        'storage_type': ['hdf5']  # might consider feather for pandas not available for vaex or dask, maybe postgre db?
    }

    def __init__(self, max_size_mb=20, storage_type='hdf5', processing_type='vaex', storage_location=STORAGE_FOLDER):
        """
        Initialize Storage and Processing Class
        :param max_size_mb: maximum size in memory in megabytes, default 20MB
        :type max_size_mb: int
        :param storage_type: method to store data that exceeds max_size, reference ALLOWED_VALUES, default hdf5
        :type storage_type: str
        :param processing_type: method to process data, reference ALLOWED_VALUES, default vaex
        :type processing_type: str
        :param storage_location: location to store data that exceeds max_size, default STORAGE_FOLDER (temp)
        :type storage_location: str
        """
        # initialized variables
        self._max_size = max_size_mb * 1024 * 1024
        self.__check_value('storage_type', storage_type)
        self._storage_type = storage_type
        self.__check_value('processing_type', processing_type)
        self._processing_type = processing_type
        self._storage_location = storage_location

        # additional class variables
        self._data = []  # list of lists [datetime, integer]
        self._storage_directory = OrderedDict()

    def __check_value(self, key, value):
        """
        Checks the values during initialization against ALLOWED_VALUES, to validate implementation is finished
        :param key: key of ALLOWED_VALUES dict
        :type key: str
        :param value: value to check in ALLOWED_VALUES
        :type value: str
        """
        valid_options = self.ALLOWED_VALUES.get(key)
        if value not in valid_options:
            raise ValueError("{0} is not a valid option for {1}. Valid options are: {2}".format(value, key,
                                                                                                valid_options))

    def get_dataframe_from_data(self, dataframe_type):
        """
        Converts internal data list of lists into dataframe
        :param dataframe_type: string representing dataframe type, options: pandas, vaex, dask
        :type: str
        :return: dataframe
        :rtype: pd.Dataframe or vx.dataframe.Dataframe or dk.dataframe
        """
        if self._data:
            dataframe = pd.DataFrame(self._data, columns=self.COLUMNS)
        else:
            dataframe = pd.DataFrame(columns=self.COLUMNS)

        dataframe.set_index(self.INDEX)
        dataframe.sort_values(self.INDEX)
        if dataframe_type == 'pandas':
            pass
        elif dataframe_type == 'vaex':
            dataframe = vx.from_pandas(dataframe)
        elif dataframe_type == 'dask':
            dataframe = dk.from_pandas(dataframe)

        return dataframe

    def add_input(self, input_data):
        """
        Adds input to data storage, sets datetime if not provided by integer, or list of integers
        :param input_data: data to input, accepts an integer, a list of integers, a list of lists, or a pandas dataframe
        :type input_data: int or list of int or list of list or pd.Dataframe
        """
        np_datetime = np.datetime64(datetime.now())
        if isinstance(input_data, pd.DataFrame):
            index_values = input_data.index.values
            input_data = input_data.values.tolist()
            # if there is an index get datetime
            if len(input_data[0]) == 1:
                input_data = [[index_values[i], input_data[i]] for i in range(0, len(index_values))]
        if isinstance(input_data, list):
            for input_value in input_data:
                if isinstance(input_value, list):
                    self.__add_input(input_value)
                else:
                    number = input_value
                    self.__add_input([np_datetime, number])
        else:
            self.__add_input([np_datetime, input_data])

    def __add_input(self, input_list):
        """
        Adds input list to internal data, exports data if it exceeds max internal size
        :param input_list: list to store
        :type input_list: list
        """
        date_time, value = input_list
        if not isinstance(date_time, np.datetime64):
            date_time = np.datetime64(date_time)
        self._data.append([date_time, value])
        if self.__is_data_at_max_size():
            self.export_data()

    def __is_data_at_max_size(self):
        """
        Returns true if data object is at max size, else false
        :return: is data at max size
        :rtype: bool
        """
        if self._max_size is not None:
            dataframe_size = sys.getsizeof(self._data)
            if dataframe_size >= self._max_size:
                print("Exceeded Memory Size: {0}".format(self._max_size))
                return True
            else:
                return False
        else:
            return False

    def export_data(self):
        """
        Exports internal data, clears data
        """
        if self._data:
            print("Exporting Data To Storage Type: {0}".format(self._storage_type))
            if self._storage_type == 'hdf5':
                self.__export_data_to_hdf5()
            else:
                raise NotImplemented("Storage type: {0} not implemented".format(self._storage_type))
            self._data = []

    def __export_data_to_hdf5(self):
        """
        Exports internal data to HDF5 file, saves datetime and filepath to storage directory
        """
        if not os.path.isdir(self._storage_location):
            os.makedirs(self._storage_location)

        # turn data into dataframe
        data_frame = self.get_dataframe_from_data(self._processing_type)

        # gets last datetime to save as filename
        last_datetime = self.get_last_datetime(data_frame)
        storage_path = os.path.join(self._storage_location, "{0}.hdf5".format(last_datetime).replace(":", "-"))

        # export dataframe
        if isinstance(data_frame, vx.dataframe.DataFrame):
            data_frame.export_hdf5(storage_path)
            print("Exported Data To: {0}".format(storage_path))
        else:
            raise NotImplemented("Big data type: {0} not implemented for export to "
                                 "hdf5".format(self._processing_type))

        # save storage path and last datetime into storage directory
        self._storage_directory[last_datetime] = storage_path

    def get_mean_and_std_deviation(self, seconds, test=False):
        """
        Gets mean and standard deviation of inputs received in the last provided seconds
        :param seconds: number of seconds to go back in storage
        :type seconds: float
        :param test: whether or not to include testing parameters
        :type test: bool
        :return: results dictionary containing minimum of 'mean' and 'std_dev'. Additional test parameters include
        'size' of dataframe, compute times 'mean_time', 'std_time' and 'size_time', and time to fetch the dataframe
        'fetch_time'
        :rtype: dict
        """
        # convert seconds to time delta and get request numpy datetime
        time_delta = timedelta(seconds=seconds)
        request_datetime = datetime.now() - time_delta
        request_np_datetime = np.datetime64(request_datetime)

        # fetch computing dataframe
        start_time = datetime.now()
        computing_dataframe = self.get_computing_dataframe(request_np_datetime)
        fetch_time = (datetime.now() - start_time).total_seconds()

        results = {}

        # calculate mean
        start_time = datetime.now()
        mean = self.get_dataframe_mean(computing_dataframe, self.EXPRESSION)
        mean_time = (datetime.now() - start_time).total_seconds()
        results['mean'] = mean

        # calculate standard deviation
        start_time = datetime.now()
        std_dev = self.get_dataframe_std_deviation(computing_dataframe, self.EXPRESSION)
        std_dev_time = (datetime.now() - start_time).total_seconds()
        results['std_dev'] = std_dev

        # add test parameters
        if test:
            # calculate dataframe size
            start_time = datetime.now()
            size = self.get_dataframe_size(computing_dataframe)
            size_time = (datetime.now() - start_time).total_seconds()
            results['size'] = size
            # set times
            results['fetch_time'] = fetch_time
            results['mean_time'] = mean_time
            results['std_dev_time'] = std_dev_time
            results['size_time'] = size_time

        return results

    def get_computing_dataframe(self, start_np_datetime):
        """
        Gets dataframe for computing, combines internal data with relevant storage dataframes. Gets all data equal to or
        greater than provided start numpy datetime
        :param start_np_datetime: the first datetime needed
        :type start_np_datetime: np.datetime64
        :return: computing dataframe
        :rtype: pd.Dataframe or vx.dataframe.Dataframe or dk.dataframe
        """
        # turn data into dataframe
        computing_dataframe = self.get_dataframe_from_data(self._processing_type)
        # get earliest date of dataframe
        first_datetime = self.get_first_datetime(computing_dataframe)

        # nothing in memory, its all in storage. go get storage
        if first_datetime is None:
            computing_dataframe = self.import_storage_into_dataframe(start_np_datetime, computing_dataframe)
        # get more from storage, earlier than in memory
        elif start_np_datetime <= first_datetime:
            storage_dataframe = self.import_storage_into_dataframe(start_np_datetime, computing_dataframe)
            computing_dataframe = self.combine_dataframes(computing_dataframe, storage_dataframe)
        # parse irrelevant data
        computing_dataframe = self.filter_dataframe_on_start_datetime(computing_dataframe, start_np_datetime)
        return computing_dataframe

    def import_storage_into_dataframe(self, start_np_datetime, dataframe):
        """
        Imports relevant data from storage directory into provided dataframe, with given start datetime. May contain
        earlier data from storage
        :param start_np_datetime: the earliest datetime requested
        :type start_np_datetime: np.datetime64
        :param dataframe: dataframe to import the data into
        :type: pd.Dataframe or vx.dataframe.Dataframe or dk.dataframe
        :return: dataframe with all data from storage from provided date until now
        :rtype: pd.Dataframe or vx.dataframe.Dataframe or dk.dataframe
        """
        if self._storage_type == 'hdf5':
            storage_list = self.get_storage_list_for_start_datetime(start_np_datetime)
            if storage_list:
                dataframe = self.make_dataframe_from_storage_list(dataframe, storage_list)
        else:
            raise NotImplemented("Importing data not implemented for storage type: {0}".format(self._storage_type))

        return dataframe

    def get_storage_list_for_start_datetime(self, start_np_datetime):
        """
        Goes through storage directory and gets all contents that newer than start np datetime
        :param start_np_datetime: the earliest datetime requested
        :type start_np_datetime: np.datetime64
        :return: list of storage files
        :rtype: list
        """
        storage_list = []
        for date_time, path in self._storage_directory.items():
            if date_time >= start_np_datetime:
                storage_list.append(path)
        return storage_list

    def import_storage_directory(self, path, clear=True):
        """
        Imports a former storage directory json file, clears existing storage directory if clear is true
        :param path: path to json storage directory
        :type path: str
        :param clear: clears storage directory in memory if true. default True
        :type clear: bool
        """
        if clear:
            self._storage_directory = OrderedDict()
        with open(path) as json_file:
            # noinspection PyTypeChecker
            # doesn't like Ordered Dictionary, is in json documentation
            storage_ordered_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            self._storage_directory += storage_ordered_dict

    def export_storage_directory(self, path, save_temp_files=True, clear=True):
        """
        Exports storage directory in memory to json file at provided path. Copies any files in temp storage to json
        folder, clears existing storage directory if clear is true
        :param path: path of json storage directory
        :type path: str
        :param save_temp_files: if true saves files in temp folder. default True
        :param clear: clears storage directory in memory if true. default True
        :type clear: bool
        """
        with open(path, 'w') as json_file:
            json.dump(self._storage_directory, json_file)
        if save_temp_files:
            directory = os.path.dirname(path)
            self.save_temp_files(directory)
        if clear:
            self._storage_directory = OrderedDict()

    def save_temp_files(self, directory):
        """
        Saves all temp files in storage directory that are in the temp folder, to the directory provided
        :param directory: folder to save temp files
        :type directory: str
        """
        for key, value in self._storage_directory.items():
            if self.STORAGE_FOLDER in value:
                filename = os.path.basename(value)
                path = os.path.join(directory, filename)
                shutil.copy(value, path)

    @staticmethod
    def get_first_datetime(dataframe, index='datetime'):
        """
        Gets first datetime from dataframe, should always be in ascending order.
        :return: first datetime
        :rtype: datetime or None
        """
        if isinstance(dataframe, vx.dataframe.DataFrame):
            datetime_list = dataframe[index].values
            if len(datetime_list) > 0:
                first_datetime = datetime_list[0]
                return first_datetime
            else:
                return None
        else:
            raise NotImplemented("Get first datetime not implemented for Dataframe type: {0}".format(type(dataframe)))

    @staticmethod
    def get_last_datetime(dataframe, index='datetime'):
        """
        Gets the last datetime from data frame, should always be in ascending order.
        :return: last datetime
        :rtype: datetime or None
        """
        if isinstance(dataframe, vx.dataframe.DataFrame):
            datetime_list = dataframe[index].values
            if len(datetime_list) > 0:
                last_datetime = datetime_list[-1]
                return last_datetime
            else:
                return None
        else:
            raise NotImplemented("Get last datetime not implemented for Dataframe type: {0}".format(type(dataframe)))

    @staticmethod
    def make_dataframe_from_storage_list(dataframe, storage_list):
        """
        Makes a dataframe from a storage list, using empty dataframe and storage list
        :param dataframe: empty dataframe
        :type dataframe: vx.dataframe.Dataframe
        :param storage_list: list of storage files
        :type storage_list: list
        :return: dataframe
        :rtype: vx.dataframe.Dataframe
        """

        if isinstance(dataframe, vx.dataframe.DataFrame):
            dataframe = vx.open_many(storage_list)
        else:
            raise NotImplemented("Export to hdf5 not implemented for Dataframe type: {0}".format(type(dataframe)))
        return dataframe

    @staticmethod
    def combine_dataframes(dataframe_1, dataframe_2):
        """
        Combines 2 dataframes of the same type
        :param dataframe_1: first dataframe
        :type dataframe_1: vx.dataframe.Dataframe
        :param dataframe_2: second dataframe
        :type dataframe_2: vx.dataframe.Dataframe
        :return: dataframe
        :rtype: vx.dataframe.Dataframe
        """
        if not type(dataframe_1) == type(dataframe_2):
            raise TypeError("Type Dataframe 1: {0} does not match Type Dataframe 2: {1}".format(type(dataframe_1),
                                                                                                type(dataframe_2)))
        elif isinstance(dataframe_1, vx.dataframe.DataFrame):
            dataframe = vx.concat([dataframe_1, dataframe_2])
        else:
            raise NotImplemented("Combine dataframes not implemented for Dataframe type: {0}".format(type(dataframe_1)))
        return dataframe

    @staticmethod
    def filter_dataframe_on_start_datetime(dataframe, start_np_datetime, index='datetime'):
        """
        Filters dataframe on start datetime returning sub dataframe
        :param dataframe: dataframe to filter
        :type dataframe: vx.dataframe.Dataframe
        :param start_np_datetime: earliest datetime
        :type start_np_datetime: np.datetime64
        :param index: column to index
        :type index: str
        :return: dataframe
        :rtype: vx.dataframe.DataFrame
        """
        if isinstance(dataframe, vx.dataframe.DataFrame):
            dataframe = dataframe[(dataframe[index] >= start_np_datetime)]
        else:
            raise NotImplemented("Filtering on start datetime not implemented for Dataframe type: "
                                 "{0}".format(type(dataframe)))
        return dataframe

    @staticmethod
    def get_dataframe_size(dataframe):
        """
        Gets dataframe size (number of rows)
        :param dataframe: dataframe to evaluate
        :type dataframe: vx.dataframe.DataFrame
        :return: size
        :rtype: int
        """
        if isinstance(dataframe, vx.dataframe.DataFrame):
            size = dataframe.count("*")
        else:
            raise NotImplemented("Get size not implemented for Dataframe type: {0}".format(type(dataframe)))
        return int(size)

    @staticmethod
    def get_dataframe_mean(dataframe, expression):
        """
        Calculates mean from dataframe with provided expression (column)
        :param dataframe: dataframe to calculate mean
        :type dataframe vx.dataframe.DataFrame
        :param expression: column or expression for mean
        :type expression: str
        :return: mean
        :rtype: float
        """
        if isinstance(dataframe, vx.dataframe.DataFrame):
            mean = dataframe.mean(expression)
        else:
            raise NotImplemented("Mean not implemented for Dataframe type: {0}".format(type(dataframe)))
        return float(mean)

    @staticmethod
    def get_dataframe_std_deviation(dataframe, expression):
        """
        Calculates standard deviation from dataframe with provided expression (column)
        :param dataframe: dataframe to calculate standard deviation
        :type dataframe vx.dataframe.DataFrame
        :param expression: column or expression for standard deviation
        :type expression: str
        :return: standard deviation
        :rtype: float
        """
        if isinstance(dataframe, vx.dataframe.DataFrame):
            std_dev = dataframe.std(expression)
        else:
            raise NotImplemented("Standard deviation not implemented for Dataframe type: {0}".format(type(dataframe)))
        return float(std_dev)


if __name__ == "__main__":
    # quick unit test, a 5 million, trillion - will exceed memory size of 20MB - runs under 1 min
    number_of_integers = 5000000
    max_integer = 1000000000000
    max_memory = 20
    print("Starting unit test. N number of Integers: {0} Max Integer: {1} Max Memory Usage: "
          "{2}MB".format(number_of_integers, max_integer, max_memory))

    # get class
    data_class = StorageAndProcessing(max_memory)

    # start input timer
    print("START: Adding random Inputs")
    input_start = datetime.now()
    # add lots of input data
    for i in range(number_of_integers):
        integer = random.randint(0, max_integer)
        data_class.add_input(integer)
    # stop input timer
    input_seconds = (datetime.now() - input_start).total_seconds()
    print("END: Adding random Inputs")
    print("Time: {0:.2f} seconds".format(input_seconds))

    # pick and random float within timer to fetch data
    m_seconds = random.random()
    if input_seconds > 1:
        m_seconds += random.randint(0, int(input_seconds))
    print("Getting M seconds of input data: {0:.2f} seconds".format(m_seconds))

    # start output timer
    print("START: Get outputs")
    output_start = datetime.now()
    # get output
    values = data_class.get_mean_and_std_deviation(m_seconds, True)
    # stop output timer
    output_seconds = (datetime.now() - output_start).total_seconds()
    print("END: Get outputs")
    print("Time: {0:.2f} seconds".format(output_seconds))

    total_seconds = input_seconds + output_seconds
    print("Total Time: {0:.2f} seconds".format(total_seconds))

    print("Results: {0}".format(values))
