from ReDReST.definitions import *
import re


class ED:
    def __init__(self, list, period, offset_hr, should_interpolate):
        self.list = list
        self.offset_hr = offset_hr
        self.period = period
        self.should_interpolate = should_interpolate

    def __getitem__(self, timeInSeconds):

        index = (self.offset_hr + int(timeInSeconds // self.period)) % len(self.list)
        # do a linear interpolation
        if self.should_interpolate:
            w = (timeInSeconds % self.period) / self.period
            if w > 0:
                next_index = 0 if index + 1 >= len(self.list) else index + 1  # wrap around
                return self.list[index] * (1 - w) + self.list[next_index] * w
            else:
                return self.list[index]
        else:
            return self.list[index]


class EDLoader:


    def __init__(self, simulation_start_time_hr):
        self.simulation_start_time_hr = simulation_start_time_hr
        self.regulation_data_file = 'regData.csv'
        self.input_data = {}
        self.load()

    def __getitem__(self, data_key):
        return self.input_data[data_key]

    def load(self):
        for input_data in EDType:
            input_options: EDOptions = input_data.value
            file = 'inputFiles\\' + input_options.filename
            print(f"reading {file}")
            with open(file,'r') as f:
                data = f.read()
            factor = input_options.factor
            period = input_options.period
            offset_hr = input_options.offset_hr
            should_interpolate = input_options.should_interpolate
            if offset_hr > self.simulation_start_time_hr:
                raise Exception(f"Invalid simulation start time. Simulation can only be started when all data is"
                                f" available. {file} is not available until {offset_hr} hour. Check constants.py")
            adjusted_offset_hr = self.simulation_start_time_hr - offset_hr
            self.input_data[input_data] = ED([float(x) * factor for x in re.split("\n|, *", data.strip())],
                                             period, adjusted_offset_hr, should_interpolate)

    def getTSD(self,data_key, start_hr, end_hr):
        lst = []
        for hour in range(int(start_hr), int(end_hr)):
            lst.append((hour * 3600, self.input_data[data_key][hour*3600]))
        return lst

if __name__ == "__main__":
    inp = EDLoader(0)
    print("Done")