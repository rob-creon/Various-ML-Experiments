import csv
import typing


class StatsCSV:
    def __init__(self, name, fields: typing.List[str]):
        self._name = name
        self._fields = fields
        self._rows = []

    def add_benchmark(self, values: typing.List[str]):
        self._rows.append(values)

    def write(self):
        with open(self._name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self._fields)
            csv_writer.writerows(self._rows)
