from ReDReST.data_utils import *

class OutputData:
    class AxisDoesNotExist(Exception):
        pass
    def __init__(self,axes):
        #list of axes with label, min and max range
        self.axes = axes
        self.axis_data = {}
        self.axis_names = set()
        for axis in axes:
            self.axis_names.add(axis[0])

    def addData(self, axis_label, data_label, data, linear_interpolate = False):
        if axis_label not in self.axis_names:
            raise OutputData.AxisDoesNotExist()
        if axis_label not in self.axis_data:
            self.axis_data[axis_label] = []
        self.axis_data[axis_label].append((data_label, data, linear_interpolate))

    def writeOutput(self,output_file,duration_hr):
        #axes = [{'axis_label'
        #data_list = [('Power',[(time_seconds,power_watts),....],
        LAST_SECOND = duration_hr*3600
        with open(output_file, 'w') as f:
            for i in range(len(self.axes)):
                axis_label = self.axes[i][0]
                axis_unit = self.axes[i][1]
                axis_min = self.axes[i][2]
                axis_max = self.axes[i][3]
                f.write("axis %s (%s), %s, %s\n"%(axis_label,axis_unit, axis_min,axis_max,))  # range from 0 to 1.5
                if axis_label not in self.axis_data or not self.axis_data[axis_label]:
                    continue
                for data_label,data, linear_interpolate in self.axis_data[axis_label]:
                    f.write("%s, " % (data_label,))
                    if not linear_interpolate:
                        data = makeStepTouplelist([(x[0]/3600,x[1]) for x in data], LAST_SECOND/3600)
                    else:
                        data = [(x[0]/3600,x[1]) for x in data] + [(LAST_SECOND/3600,data[-1][1])]

                    f.write(','.join([str(d[0]) + ',' + str(d[1]) for d in data]) + '\n')
