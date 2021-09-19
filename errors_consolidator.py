import json
import os

from collections import OrderedDict


def json_orchestrate(predicator_length, forecast_length, category, errors_dict):
    if category not in ['arima', 'markov']:
        raise Exception('Category must be one of (arima | markov)')

    this_errors_category = category
    this_errors_space = f'{predicator_length}p{forecast_length}'
    this_errors_dict = OrderedDict(errors_dict)

    errors_file_path = 'errors_parallel.json'

    # Create file if it doesn't exist
    if not os.path.exists(errors_file_path):
        temp_file = open(errors_file_path, 'w')
        json.dump({}, temp_file)
        temp_file.close()

    # Load data from the file
    errors_file = open(errors_file_path, 'r')
    old_data = OrderedDict(json.load(errors_file))
    errors_file.close()

    # Orchestrate
    new_errors_dict = OrderedDict()
    new_errors_dict[this_errors_space] = OrderedDict({category: this_errors_dict})

    # Update the old data
    new_data = old_data.update(new_errors_dict)

    # Dump new data to the file
    with open(errors_file_path, 'w') as errors_file:
        json.dump(new_data, errors_file)