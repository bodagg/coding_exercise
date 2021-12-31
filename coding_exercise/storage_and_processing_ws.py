#!/usr/bin/env python

"""
Quick Webservice for Running Storage and Processing

Repository:
coding_exercise

History:
- 12/29/2021 - CSM - original creation
"""

import cherrypy
import json
from coding_exercise.storage_and_processing import StorageAndProcessing

S_and_P = StorageAndProcessing()


class StorageAndProcessingWebservice:
    """
    Class to contain webservice
    """
    def __int__(self, test=False):
        """
        Initialize web service class
        :param test: return test parameters if true, default False
        :type test: bool
        """
        self._test = test

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def add_input(self):
        """
        Adds input to storage and processing class, expects json with input param
        :return: json with status
        :rtype: json
        """
        input_json_data = cherrypy.request.json
        input_dict_data = json.load(input_json_data)
        try:
            input_integer = input_dict_data.get('input')
            S_and_P.add_input(input_integer)
        except Exception as e:
            return json.dumps({'status': 500, 'exception': e})
        else:
            return json.dumps({'status': 200})

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def get_output(self):
        """
        Gets requested output of mean and standard deviation
        :return: json with results and status
        :rtype: json
        """
        input_json_data = cherrypy.request.json
        input_dict_data = json.load(input_json_data)
        try:
            input_seconds = input_dict_data.get('seconds')
            results = S_and_P.get_mean_and_std_deviation(input_seconds, self._test)
        except Exception as e:
            return json.dumps({'status': 500, 'exception': e})
        else:
            results['status'] = 200
            return json.dumps(results)


if __name__ == '__main__':
    config = {'server.socket_host': '0.0.0.0'}
    cherrypy.config.update(config)
    cherrypy.quickstart(StorageAndProcessingWebservice())
