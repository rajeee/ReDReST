import numpy
from scipy import interpolate
import itertools


def minAvgMax(Data):
    return [min(Data), sum(Data) / len(Data), max(Data)]

def calculate_cycle_times(powers):
    f_min = list()
    f_max = list()
    f_avg = list()
    s_min = list()
    s_max = list()
    s_avg = list()
    cycles = list()
    for power in powers:
        times = numpy.array([t[0] for t in power])
        offset_time = times[1:]
        preset_time = times[:-1]
        intervals = offset_time-preset_time
        second_intevals = numpy.array([])
        first_intervals = numpy.array([])
        if power and power[0] and power[0][1] > 0:
            first_intervals = intervals[0::2]
            second_intevals = intervals[1::2]
        else:
            first_intervals = intervals[1::2]
            second_intevals = intervals[0::2]
        if len(second_intevals)==0:
            second_intevals = [0]
        if len(first_intervals)==0:
            first_intervals = [0]
        f_min.append(min(first_intervals))
        f_max.append(max(first_intervals))
        f_avg.append(sum(first_intervals)/len(first_intervals))
        s_min.append(min(second_intevals))
        s_max.append(max(second_intevals))
        s_avg.append(sum(second_intevals)/len(second_intevals))
        cycles.append(len(first_intervals))

    return {'min_on': minAvgMax(f_min),'avg_on':minAvgMax(f_avg),'max_on': minAvgMax(f_max),
            'min_off':minAvgMax(s_min),'avg_off':minAvgMax(s_avg), 'max_off':minAvgMax(s_max),'Cycles':minAvgMax(cycles)}

def calculate_cost(power,price):
    power = list(power)
    unzip_price = zip(*price)
    price_t = next(unzip_price)
    price_p = next(unzip_price)
    price_f = interpolate.interp1d(price_t,price_p)
    before_power = power[0]
    before_price = price_f(power[0][0])
    cost = 0
    energy = 0
    for p in power:
        time_diff = p[0] - before_power[0]
        try:
            cur_price = price_f(p[0])
        except ValueError:
            continue  #skip this power

        price_avg = (cur_price + before_price)/2
        power_avg = (p[1] + before_power[1]) / 2
        energy += power_avg*time_diff
        cost += power_avg*time_diff*price_avg
        before_power = p
        before_price = cur_price

    return {'cost':cost/(60*100),'energy':energy/(60)}

def makeStepTouplelist(data: list,last_time):
    if not data:
        return [(0,0)]
    data_zip = zip(*data)
    t = next(data_zip)
    p = next(data_zip)
    old_time = numpy.array(t)
    old_p = numpy.array(p)
    old_list = numpy.vstack((old_time, old_p)).transpose().tolist()
    new_time = numpy.append(old_time[1:], last_time) - 0.1/3600
    new_list = numpy.vstack((new_time, old_p)).transpose().tolist()
    final = numpy.array(list(itertools.chain(*zip(old_list, new_list))))
    return zip(final[:, 0], final[:, 1])

def getComfortViolationIndex(temperaturesList: list, desiredTemps: list, deadband: float, startTime: float=None, endTime: float=None):
    degreeHoursList = [[]]*len(temperaturesList)
    i=0
    for temperatures in temperaturesList:
        desiredTemp = desiredTemps[i]
        degreeHours = 0
        lastTime = None
        lastTemp = None
        avgTemp = None
        for time,temp in temperatures:
            if lastTime:
                hourDiff = (time - lastTime) / 60.0
                avgTemp = (temp + lastTemp) / 2

            if avgTemp and (startTime is None or time >= startTime) and (endTime is None or time <= endTime):
                if avgTemp >  desiredTemp + deadband or avgTemp < desiredTemp - deadband:
                        degreeHours += (abs(avgTemp-desiredTemp)-deadband) * hourDiff
            lastTime, lastTemp = time, temp
        degreeHoursList[i] = (i,degreeHours)
        i += 1

    return  degreeHoursList


def delayed_coeff(ref,res,delay):
    left = ref[:-delay] if delay else ref
    right = res[delay:]
    score = numpy.corrcoef(left,right)[0,1]
    return score

def TenSecDiff(ref,res):
    N = len(ref)
    diff = []
    for i in range(0,N,5): #five used for 10-sec avg because time step is 2 seconds for ref signal
        avgRef = sum(ref[i:i+5])/5
        avgRes = sum(res[i:i+5])/5
        diff.append(abs(avgRef-avgRes)*5)
    return diff

def mileage(ref):
    ref = numpy.array(ref)
    mil = ref[1:]-ref[:-1]
    return sum(abs(mil))


def calculatePerformanceScore(mid_point_power,reference_power,response_power,ignore_interpolation=False):


    if len(reference_power)==0 or len(response_power)==0:
        return {"ps": -1, "ds": -1, "cs": -1, "ts": -1,
                "avg_ref": -1}
    if not ignore_interpolation:
        unzip_res = zip(*response_power)
        res_t = list(next(unzip_res)) #get the list of timestamps of response power
        res_p = list(next(unzip_res))
        res_f = interpolate.interp1d(res_t,res_p,kind='zero') #create interpolation function for response power
        unzip_ref = zip(*reference_power)
        ref_t = list(next(unzip_ref)) #get list of timestamps for reference power
        ref_p = list(next(unzip_ref))
        res_p = []
        for t in ref_t:
            if t > res_t[-1]:
                #if reference time greater than the maximum response time, assume response remained the same after that\
                res_p.append(res_p[-1])
            else:
                res_p.append(res_f(t)) #get the response power for the same timestamps as reference power
        res_p = numpy.array(res_p)
        ref_p = numpy.array(ref_p)
    else:
        res_p = response_power[:,1]
        ref_p = reference_power[:,1]

    scores = [delayed_coeff(ref_p,res_p,delay) for delay in range(5*60)] #get co-relation coeficient for a range of delays
    correlation_score = max(scores) #the corelation score is the max out of it
    best_delay = numpy.argmax(scores)
    delay_score = abs((best_delay-5*60)/(5*60))
    avg_ref = sum([abs(p-mid_point_power) for p in ref_p])/len(ref_p)
    print("AVG ref power is: " + str(avg_ref))
    diffs = TenSecDiff(ref_p,res_p)
    #old_avg_error = sum([abs(res_p[i] - ref_p[i]) / avg_ref for i in range(len(ref_p))]) / len(ref_p)
    avg_error = sum(diffs)/(avg_ref*len(ref_p))
    precision_score = 1-avg_error
    total_score = (precision_score+delay_score+correlation_score)/3

    return {"ps":precision_score,"ds":delay_score,"cs":correlation_score,"ts":total_score,"avg_ref":avg_ref}

def calculateMultiHourPerformanceScore(hourly_avg,reference_power,response_power):
    resp = numpy.array(response_power)
    refp = numpy.array(reference_power)
    res_f = interpolate.interp1d(resp[:,0],resp[:,1], kind='zero')
    vres_f = numpy.vectorize(res_f)
    interpolated_response = res_f(refp[:,0])
    iresp = numpy.vstack((refp[:,0],interpolated_response)).transpose()
    begin_hour = int(refp[0,0]/60)
    end_hour = int(refp[-1,0]/60)
    lastpos = 0
    total_score = [-1]*24
    precision_score = [-1]*24
    delay_score = [-1]*24
    corelation_score = [-1]*24
    for i in range(1,24):
        if hourly_avg[i-1] != 0:
            end = i * 60
            pos = numpy.searchsorted(refp[:, 0], end)
            if pos == 0:
                pos = refp.shape[0]
            hourlyPFS = calculatePerformanceScore(hourly_avg[i-1],refp[lastpos:pos,:],iresp[lastpos:pos,:],ignore_interpolation=True)
            lastpos = pos
        else:
            hourlyPFS = {"ps": -1, "ds": -1, "cs": -1, "ts": -1,
             "avg_ref": -1}
        precision_score[i-1]= hourlyPFS['ps']
        total_score[i-1] = hourlyPFS['ts']
        delay_score[i-1] = hourlyPFS['ds']
        corelation_score[i-1] = hourlyPFS['cs']


    return total_score, precision_score, corelation_score, delay_score



