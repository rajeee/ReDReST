from __future__ import annotations

import simpy
ENV = simpy.Environment()
import random
import math
random.seed(21)
from ReDReST.readExternalData import EDLoader, ED
from collections import namedtuple
import solver
from typing import Dict, List, Tuple, Set, Optional
from abc import ABCMeta, abstractmethod, ABC

EPS = 1e-6 #the epsilon for numerical tolernace
GALPCF = 7.4805195 #// gal/cf
RHOWATER = 62.4 # // lb / cf
Cp = 1 #Btu/lbm-F for water #specific heat capacity
BTUPHPW	= 3.4120 # BTUPH/W
BTUPHPKW = (1e3 * 3.4120)#        // BTUPH/kW
KWPBTUPH = (1e-3/BTUPHPW) #   // kW/BTUPH
MWPBTUPH = (1e-6/BTUPHPW) #   // MW/BTUPH
ROUNDOFF = 1e-6           # // numerical accuracy for zero in float comparisons

THERMAL_PARAMETERS = namedtuple('THERMAL_PARAMETERS', 'A1, A2, A3, A4, r1, r2, a, b, c, d, g, hvac_power')

SIMULATION_DAY = 214


inputData = EDLoader(simulation_start_time_hr=(SIMULATION_DAY - 1) * 24)



def clip(variable, low, high):
    if variable > high:
        return high
    elif variable < low:
        return low
    else:
        return variable

from ReDReST.definitions import *


class DRAuthority:
    def __init__(self):
        self.env = ENV
        self.aggregator_set = set()
        self.DREvents: List[Tuple[float, float, DRType, Optional[ED]]] = []
        self.number_of_aggregators = 0

    def registerAggregator(self, dr_aggregator: DRAggregator):
        if dr_aggregator not in self.aggregator_set:
            self.number_of_aggregators +=1
            self.aggregator_set.add(dr_aggregator)

    def createDREvent(self, start_hour: float, end_hour: float, dr_type:DRType, regulation_data: ED = None):
        if start_hour*3600 <= self.env.now:
            raise Exception("DR event can only be set for future events")
        if self.DREvents and start_hour<self.DREvents[-1][1]:
            raise Exception("The start hour cannot be less than the end hour of last event entered")
        if dr_type == DRType.Regulation and regulation_data is None:
            raise Exception("Regulation DR requires supplying external regulation data")

        self.DREvents.append((start_hour*3600,end_hour*3600, dr_type, regulation_data))

    def start(self):
        self.env.process(self.processDR())

    def processDR(self):
        for next_event_start,next_event_end, dr_type, dr_data in self.DREvents:
            print(f"DRAuthority: Next up Event: {dr_type} at {next_event_start} until {next_event_end}")
            yield self.env.timeout(next_event_start-self.env.now)
            for aggregator in self.aggregator_set:
                aggregator.startDR(next_event_end, dr_type, dr_data)


class DRAggregator:
    class HouseNotRegistered(Exception):
        pass

    def __init__(self, dr_authority: DRAuthority):

        self.dr_authority = dr_authority
        self.dr_authority.registerAggregator(self)
        self.total_exact_power = 0
        self.running_rated_power = 0
        self.house_powers: Dict[House, float] = {}
        self.house_set: Set[House] = set()
        self.power_record: List[Tuple[float,float]] = []
        self.DR_limit_record: List[Tuple[float, float]] = []
        self.env = ENV
        self.number_of_houses = 0
        self.inDR = False
        self.DR_end_time = 0
        self.DR_event_start : simpy.Event = self.env.event()
        self.DR_event_end: simpy.Event = self.env.event()
        self.restrike_limit_duration = 3*3600
        #env.process(self.load_shape_driver())


    def register(self, house: House):

        if house not in self.house_set:
            self.number_of_houses +=1
            self.house_set.add(house)
            self.house_powers[house] = 0
            self.changePower(house, 0)

    def changePower(self, house: House, value: float):
        if house not in self.house_set:
            raise DRAggregator.HouseNotRegistered()

        self.total_exact_power -= self.house_powers[house] #remove the old power from the count
        self.house_powers[house] = value #update the current power of appliance
        self.total_exact_power += value #add it to the total

        if self.power_record and self.power_record[-1][0] == self.env.now:
            self.power_record.pop() #if the last entry is also for the same time, remove it and update it

        self.power_record.append((self.env.now, self.total_exact_power))
        # if self.env.now // 3600 > 55:
        #     print("I am Washer %s: Time: %s, and I changing power during DR event" % (appliance.id, self.env.now / 3600))

    def startDR(self,end_time: float, dr_type: DRType, dr_data: ED):
        max_hvac_load = 0
        max_water_heater_load = 0
        app_lists: Dict[ApplianceType, List[GenericAppliance]] = {}
        max_loads: Dict[ApplianceType, float] = {}

        for app_type in ApplianceType:
            app_lists[app_type] =  [house.appliances[app_type] for house in self.house_set if app_type in house.appliances]
            max_loads[app_type] = sum([house.appliances[app_type].rated_power for house in self.house_set if app_type in house.appliances])


        for house in self.house_set:
            for app_type in [ApplianceType.Dryer, ApplianceType.Washer, ApplianceType.Dishwasher]:
                if app_type in house.appliances:
                    house.appliances[app_type].startDR(end_time=end_time)

            for app_type in [ApplianceType.HVAC, ApplianceType.WaterHeater]:
                if app_type in house.appliances:
                    house.appliances[app_type].startDR(end_time=end_time+self.restrike_limit_duration)

        hvac_limit = max_loads[ApplianceType.HVAC] * 0.16
        hvac_restrike_limit = max_loads[ApplianceType.HVAC] * 0.3
        water_heater_limit =  max_loads[ApplianceType.WaterHeater] * 0.03
        water_heater_restrike_limit = max_loads[ApplianceType.WaterHeater] * 0.14
        print(f"DR HVAC limit {hvac_limit} and restrike limit {hvac_restrike_limit}")
        print(f"DR Water heater limit {water_heater_limit} and restrike limit {water_heater_restrike_limit}")
        self.env.process(self.handleLoadReductionDR(ApplianceType.HVAC, 5 * 60, end_time,hvac_limit ,
                                                    restrike_limit_level=hvac_restrike_limit,
                                                    restrike_limit_duration=self.restrike_limit_duration))
        self.env.process(self.handleLoadReductionDR(ApplianceType.WaterHeater, 5 * 60, end_time, water_heater_limit,
                                                    restrike_limit_level=water_heater_restrike_limit,
                                                    restrike_limit_duration=5*3600))

        self.env.process(self.endAppliancesDR(end_time+self.restrike_limit_duration-self.env.now, app_lists[ApplianceType.HVAC]))
        self.env.process(self.endAppliancesDR(end_time + 5*3600-self.env.now, app_lists[ApplianceType.WaterHeater]))

        self.env.process(self.endAppliancesDR(end_time-self.env.now, app_lists[ApplianceType.Dishwasher]))
        self.env.process(self.endAppliancesDR(end_time-self.env.now, app_lists[ApplianceType.Dryer]))
        self.env.process(self.endAppliancesDR(end_time-self.env.now, app_lists[ApplianceType.Washer]))

    def endAppliancesDR(self,delay, appliancesList: List[GenericAppliance]):
        yield self.env.timeout(delay)
        for appliance in appliancesList:
            appliance.endDR()

    def handleLoadReductionDR(self, app_type: ApplianceType, control_interval, end_time, limit_level, restrike_limit_level = 0.0, restrike_limit_duration = 0.0 ):
        start_time = self.env.now
        while self.env.now < end_time + restrike_limit_duration:
            if self.env.now == 63900:
                print("Gotxa")
            load_limit = limit_level if self.env.now <= end_time else restrike_limit_level
            eligible_appliances: List[GenericAppliance] = []
            current_load = 0
            to_run_appliances: List[GenericAppliance] = []
            max_possible_load = 0
            must_turn_off_appliances: List[GenericAppliance] = []
            must_turn_on_appliances: List[GenericAppliance] = []
            for house in self.house_set:
                if house.appliances[app_type].running and not house.appliances[app_type].can_turn_off():
                    current_load += house.appliances[app_type].rated_power

                if not house.appliances[app_type].running and house.appliances[app_type].must_turn_on():
                    must_turn_on_appliances.append(house.appliances[app_type])
                    current_load += house.appliances[app_type].rated_power

                if house.appliances[app_type].can_turn_on():
                    eligible_appliances.append(house.appliances[app_type])


                elif house.appliances[app_type].running and house.appliances[app_type].must_turn_off():
                    must_turn_off_appliances.append(house.appliances[app_type])

            eligible_appliances = sorted(eligible_appliances,key=lambda app: app.get_turn_on_priority_weight(), reverse=True)

            to_run_appliances = must_turn_on_appliances #start with must turn on appliances

            if current_load < load_limit:
                while current_load < load_limit and eligible_appliances:
                    next_hvac = eligible_appliances.pop()
                    to_run_appliances.append(next_hvac)
                    current_load += next_hvac.rated_power

            for appliance in to_run_appliances:
                if not appliance.running:
                    if not appliance.external_state_change_event.triggered:
                        appliance.external_state_change_event.succeed(value=1)
                    else:
                        appliance.external_state_change_event._value = 1

            for appliance in eligible_appliances + must_turn_off_appliances:
                if appliance.running and appliance.can_turn_off():
                    appliance.external_state_change_event.succeed(value=0)

            yield self.env.timeout(control_interval)
            #print(f"{app_type} DR is in Progress {self.env.now}. Progress: {(self.env.now-start_time)/(end_time-start_time)}")

    def handleRegulationDR(self, app_type: ApplianceType, control_interval, end_time, midPointLevel, regulation_amount, regulation_data):
        pass

class ApplianceDriver:
    class ApplianceNotRegistered(Exception):
        pass

    def __init__(self, app_type: ApplianceType, load_shape_data: ED = None):

        self.appliance_ype = app_type
        self.total_exact_power = 0
        self.running_rated_power = 0
        self.appliance_powers: Dict[GenericAppliance,float] = {}
        self.appliance_status: Dict[GenericAppliance,bool] = {}
        self.appliance_set: Set[GenericAppliance] = set()
        self.power_record: List[Tuple[float,float]] = []
        self.loadshape_record: List[Tuple[float,float]] = []
        self.status_record: List[Tuple[float,int]] = []
        self.on_count: int = 0
        self.env = ENV
        self.number_of_appliances = 0
        self.load_shape_data = load_shape_data
        if load_shape_data:
            self.env.process(self.load_shape_driver())


    def register(self,appliance):
        if appliance not in self.appliance_set:
            self.number_of_appliances +=1
            self.appliance_set.add(appliance)
            self.appliance_powers[appliance] = 0
            self.appliance_status[appliance] = False
            self.changePower(appliance,0)
            self.changeStatus(appliance,0)

    def changeStatus(self,appliance,status):
        if appliance not in self.appliance_set:
            raise ApplianceDriver.ApplianceNotRegistered()

        if (status == 1 and self.appliance_status[appliance] == 0) or \
            status == 0 and self.appliance_status[appliance] == 1:
            self.appliance_status[appliance] = status
            self.on_count += 1 if status == 1 else -1
            self.running_rated_power += appliance.rated_power if status == 1 else -1*appliance.rated_power

            if self.status_record and self.status_record[-1][0] == self.env.now:
                self.status_record.pop()
            self.status_record.append((self.env.now,self.on_count))

    def changePower(self,appliance,value):
        if appliance not in self.appliance_set:
            raise ApplianceDriver.ApplianceNotRegistered()

        self.total_exact_power -= self.appliance_powers[appliance] #remove the old power from the count
        self.appliance_powers[appliance] = value #update the current power of appliance
        self.total_exact_power += value #add it to the total

        if self.power_record and self.power_record[-1][0] == self.env.now:
            self.power_record.pop() #if the last entry is also for the same time, remove it and update it

        self.power_record.append((self.env.now, self.total_exact_power))
        # if self.env.now // 3600 > 55:
        #     print("I am Washer %s: Time: %s, and I changing power during DR event" % (appliance.id, self.env.now / 3600))

    def load_shape_driver(self):
        while True:
            required_watts_now = self.load_shape_data[self.env.now]*self.number_of_appliances
            self.loadshape_record.append((self.env.now,required_watts_now))
            sortedAppliances = sorted(self.appliance_powers.keys(), key = lambda app: app.load, reverse=True)
            additional_started_power = 0
            if self.running_rated_power < required_watts_now + additional_started_power:
                for appliance in sortedAppliances:
                    # if self.env.now > 5*3600:
                    #     print("Arrive")
                    if appliance.running or appliance.inDR or not appliance.can_turn_on():
                        continue
                    #if appliance.rated_power < required_watts_now - (self.total + additional_started_power):
                    if not appliance.inDR and not appliance.external_state_change_event.triggered:
                        appliance.external_state_change_event.succeed(value=1)
                    additional_started_power += appliance.rated_power
                    if self.running_rated_power + additional_started_power > required_watts_now:
                        break

            yield self.env.timeout(60)


class GenericAppliance(metaclass=ABCMeta):
    def __init__(self, app_id, app_type:ApplianceType, house: House, app_driver: ApplianceDriver = None):
        self.app_id = app_id
        self.app_type = app_type
        self.house = house
        self.power_record: List[Tuple[float,float]] = []
        self.loading_record: List[Tuple[float,float]] = []
        self.running = False
        self.inDR = False
        self.DR_end_time = 0
        self.env = ENV
        if app_driver:
            self.app_driver: ApplianceDriver = app_driver
            self.app_driver.register(self)
        self.external_state_change_event = self.env.event()
        self.last_run_start_time = -float('inf')
        self.last_run_stop_time = -float('inf')

        self.env.process(self.run())

    def startDR(self,end_time):
        self.inDR = True
        self.DR_end_time = end_time

    def endDR(self):
        self.inDR = False


    @abstractmethod
    def get_turn_on_priority_weight(self):
        raise NotImplementedError

    @abstractmethod
    def can_turn_off(self):
        raise NotImplementedError

    @abstractmethod
    def can_turn_on(self):
        raise NotImplementedError

    @abstractmethod
    def must_turn_off(self):
        raise NotImplementedError

    @abstractmethod
    def must_turn_on(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def rated_power(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def runCycles(self):
        raise NotImplementedError

    def next_event_time(self):
        return 60 #default to every 60 seconds


    def changePower(self,power):
        self.app_driver.changePower(self, power)
        self.house.changePower(self,power)
        self.power_record.append((self.env.now, power))

    def changeStatus(self,status):
        if self.running != status:
            if status == True:
                self.last_run_start_time = self.env.now
            else:
                self.last_run_stop_time = self.env.now

            self.running = status
            self.app_driver.changeStatus(self, status)

    @abstractmethod
    def updateInternalStates(self, time=None):
        #Advance the internal state variables by the provided time
        raise NotImplementedError

        #self.load = self.load - 1 if self.load > 1 else 0
        #self.loading_record.append((self.env.now, self.load))
        #self.temp += (self.heating_capacity / 60) / (self.heat_capacity)
        #self.temp_record.append((self.env.now, self.temp))

    @abstractmethod
    def applyControlLogic(self, external_state_request):
        #If required, based on the current internal state variables, change the appliance status/working
        raise NotImplementedError

    def run(self):
        while True:
            dt = self.next_event_time()
            if math.isnan(dt) or dt < 10:
                dt = 10
            elif dt >60:
                dt = 60
            before = self.env.now
            timeout_event = simpy.events.Timeout(self.env,dt)
            yielded_event_values = yield self.external_state_change_event | timeout_event
            after = self.env.now
            self.updateInternalStates(time=after-before)
            external_state_request = None
            if self.external_state_change_event in yielded_event_values:
                external_state_request = yielded_event_values[self.external_state_change_event]
                self.external_state_change_event = self.env.event()  # create another fresh one

            self.applyControlLogic(external_state_request)

class WasherLikeAppliance(GenericAppliance, ABC):

    def __init__(self,*args,**kwargs):
        super(WasherLikeAppliance, self).__init__(*args, **kwargs)
        self.runCyclesData = [(2*60,6),(15*60,500),(2*60,600),(2*60,6),(9*60,500),(2*60,600)]
        self.load = random.uniform(0, 1)
        self.loading_record.append((self.env.now, self.load))
        self.loads_per_week = 3
        total_run_time = 0
        for stage in self.runCyclesData:
            total_run_time += stage[0]
        self.total_run_time = total_run_time
        self.current_power = 0

    def __repr__(self):
        return f"{self.app_type} {self.app_id}, Temp: {self.load}, Status: {self.running}"


    def get_turn_on_priority_weight(self):
        return self.load

    def next_event_time(self):
        if not self.running:
            return 5*60 #when not running update every 5 minute
        else:
            offset = self.env.now - self.last_run_start_time
            if offset < 0 or offset > self.total_run_time:
                return 5*60
            else:
                time_cum_sum = 0
                power = 0
                for stage in self.runCycles:
                    power = stage[1]
                    time_cum_sum += stage[0]
                    if time_cum_sum > offset:
                        break

            return time_cum_sum - offset


    def applyControlLogic(self, external_state_request=None):

        if external_state_request is not None:
            if external_state_request == 1 and self.running == 0:
                self.last_run_start_time = self.env.now
                self.changeStatus(True)
                self.changePower(self.runCyclesData[0][1]) #apply the first power
            if external_state_request == 0 and self.running == 1:
                self.last_run_stop_time = self.env.now
                self.changeStatus(False)
                self.changePower(0)
        else:
            if self.env.now - self.last_run_start_time >= self.total_run_time and self.running:
                self.last_run_stop_time = self.env.now
                self.changeStatus(False)
                self.changePower(0)

    def getCurrentPower(self, time):
        #returns the power consumption from the cycles data using offset
        offset = time - self.last_run_start_time
        if offset < 0 or offset > self.total_run_time:
            return 0
        else:
            time_cum_sum = 0
            power = 0
            for stage in self.runCycles:
                power = stage[1]
                time_cum_sum += stage[0]
                if time_cum_sum > offset:
                    break
        return power

    def can_turn_off(self):
        return not self.running #running appliance can't be turned off

    def can_turn_on(self):
        return self.env.now - self.last_run_stop_time > 5*3600 #only if it's been 5 hours since last stop

    def must_turn_off(self):
        return False

    def must_turn_on(self):
        return self.load > 5 #like upper limit for the load

    @property
    def loading_per_second(self):
        return self.loads_per_week / (7 * 24 * 60 * 60)

    @property
    def runCycles(self):
        return self.runCyclesData

    @property
    def max_load_for_dr(self):
        return 2

    @property
    def load(self):
        return self._load

    @load.setter
    def load(self, value):
        self._load = value

    def updateInternalStates(self, time=None):

        if self.running:
            new_power = self.getCurrentPower(self.env.now)
            if new_power != self.current_power:
                self.changePower(new_power)
                self.current_power = new_power
            self.load = self.load - 1*time/self.total_run_time if self.load > 1 else 0
        else:
            self.load += self.loading_per_second*time

        self.loading_record.append((self.env.now, self.load))


class Washer(WasherLikeAppliance):

    @property
    def rated_power(self):
        return 500


class Dryer(WasherLikeAppliance):

    def __init__(self, *args,**kwargs):
        super(Dryer, self).__init__(*args, **kwargs)
        self.runCyclesData = [(18*60,3000),(2*60,226),(5*60,3000),(2.5*60,226),(2.5*60,300),(10*60,226)]

    @property
    def rated_power(self):
        return 2000

class Dishwasher(WasherLikeAppliance):
    def __init__(self, *args,**kwargs):
        super(Dishwasher, self).__init__(*args, **kwargs)
        self.runCyclesData = [(12*60,250),(11*60,1150),(27*60,250),(12*60,1150),(10*60,250),(5*60,0),(20*60,600)]
        self.loads_per_week = 4

    @property
    def rated_power(self):
        return 800

class WaterHeater(GenericAppliance):

    def __init__(self, *args, **kwargs):

        super(WaterHeater,self).__init__(*args,**kwargs)


        self.tank_volume = random.uniform(40,80)
        self.temp_record = []
        self.usage_fraction: ED = inputData[EDType.WaterUsageFraction]
        #self.temp_setpoint = random.uniform(100,160)
        self.temp_setpoint = 140
        self.deadband = 10
        self.temp = random.uniform(self.temp_setpoint - self.deadband,self.temp_setpoint)
        self.water_use_per_day = 1.12 #random.uniform(0.3,0.6)*self.tank_volume
        self.design_inlet_temp = 45
        self.temp_drop_per_second = self.water_use_per_day*(self.temp_setpoint-self.design_inlet_temp)/(24*60*60)
        self.temp_record.append((self.env.now,self.temp))
        self.running_hours_per_day = 2

        #Cw = tank_volume / GALPCF * RHOWATER * Cp; // [Btu / F]  # water heater heat capacity

        self.heat_capacity = self.tank_volume/GALPCF * RHOWATER * Cp #[Btu / F]
        self.heating_capacity = self.rated_power * BTUPHPW # BTU/hr

    def get_turn_on_priority_weight(self):
        diff = self.temp_setpoint-self.temp
        return -diff

    def can_turn_on(self):
        max_temp = 160 if self.inDR else self.temp_setpoint
        return self.temp < max_temp

    def can_turn_off(self):
        max_temp = 160 if self.inDR else self.temp_setpoint
        return self.temp > max_temp

    def must_turn_on(self):
        min_temp = 100 if self.inDR else self.temp_setpoint - 10
        return self.temp < min_temp

    def must_turn_off(self):
        min_temp = 100 if self.inDR else self.temp_setpoint - 10
        return self.temp > min_temp

    def applyControlLogic(self, external_state_request=None):
        if external_state_request is not None:
            if external_state_request == 1 and self.running == 0:
                self.last_run_start_time = self.env.now
                self.changeStatus(True)
                self.changePower(self.rated_power) #apply the first power
            if external_state_request == 0 and self.running == 1:
                self.last_run_stop_time = self.env.now
                self.changeStatus(False)
                self.changePower(0)
        else:
            if self.env.now - self.last_run_start_time >= 2*60 and self.running and self.temp > self.temp_setpoint:
                self.last_run_stop_time = self.env.now
                self.changeStatus(False)
                self.changePower(0)

        if not self.inDR:
            if self.temp > self.temp_setpoint and self.running == 1:
                self.changeStatus(0) #the temperature too high; must turn off
                self.changePower(0)
            elif self.temp < self.temp_setpoint -  self.deadband and self.running == 0:
                self.changeStatus(1) #temperature too low; must turn off
                self.changePower(self.runCycles[0][1])

    @property
    def rated_power(self):
        return 4500

    @property
    def runCycles(self):
        run_time = 3600 * (self.temp_setpoint - self.temp) * self.heat_capacity / self.heating_capacity
        return [(run_time,self.rated_power)]

    @property
    def loading_per_second(self):
        return -self.temp_drop_per_second

    @property
    def max_load_for_dr(self):
        return 100

    @property
    def load(self):
        return self.temp_setpoint - self.temp

    @load.setter
    def load(self, load):
        self.temp = self.temp_setpoint - load

    def updateInternalStates(self, time: float=None):
        if self.running:
            self.temp += (self.heating_capacity * time / 3600) / (self.heat_capacity)
        elif not self.running:
            self.temp -= (self.temp_drop_per_second*self.usage_fraction[self.env.now]) * time
        self.temp_record.append((self.env.now, self.temp))

class HVAC(GenericAppliance):
    def __init__(self, app_id, app_type:ApplianceType, house: House, app_driver: ApplianceDriver=None, **kwargs):

        super(HVAC, self).__init__(app_id, app_type, house, app_driver)

        self.temp_record =  []

        RANDOMIZE_HOUSE = True
        RANDOMIZE_SETPOINTS = False
        AC_BASE_SETPOINT = 77
        UPPER_BOUNDARY_OFFSET = 2
        LOWER_BOUNDARY_OFFSET = 2
        AIR_HEAT_CAPACITY = 0.0195
        HUMAN_HEAT = 300

        HVAC_POWER = 3.8  # kW
        HVAC_COOLING_CAPACITY = 12000  # Btu/h
        AC_MIN_ON_TIME = 2 * 60  # minimum seconds for which the compressor must run before can be turned off
        AC_MIN_OFF_TIME = 3 * 60  # minimum seconds for which the compressor must be turned off before can be turned back on
        AC_PANIC_MARGIN = 5  # Seconds before which compressor are forced turned on or forced turned off to prevent hitting the boundary
        AC_PANIC_SAFETY_FACTOR = 1  # An HVAC won't be turned on or off if by the end of it's minimum on/off time, the remaiining time to
        # boundary would be less than  PANIC_SAFETY_FACTOR*PANIC_MARGIN
        AC_DT = 2  # two second time interval

        basic_cooling_schedule = [[0, AC_BASE_SETPOINT], [8 * 60, AC_BASE_SETPOINT], [23 * 60, AC_BASE_SETPOINT],
                                  [23.5 * 60, AC_BASE_SETPOINT]]

        COOLING_MODE = 1
        HEATING_MODE = 2

        OPERATION_MODE = COOLING_MODE




        self.floor_area = 2457 #sf
        self.cooling_cop = 3.5
        self.aspect_ratio = 1.5
        if RANDOMIZE_HOUSE:
            self.floor_area = random.gauss(2200, 400)
            self.aspect_ratio += random.uniform(-0.3, 0.3)

        if 'floor_area' in kwargs:
            self.floor_area = kwargs['floor_area']

        # self.floor_area = 1930.0
        self.ceiling_height = 8.0
        self.number_of_stories = 1
        self.gross_wall_area = 2.0 * self.number_of_stories * (
                self.aspect_ratio + 1.0) * self.ceiling_height * ((self.floor_area / self.aspect_ratio /
                                                                   self.number_of_stories) ** 0.5)
        self.window_wall_ratio = 0.15
        self.window_roof_ratio = 0.0
        self.number_of_doors = 4
        self.interior_exterior_wall_ratio = 1.5
        self.exterior_wall_fraction = 1.0
        self.exterior_ceiling_fraction = 1.0
        self.exterior_floor_fraction = 1.0
        self.window_exterior_transmission_coefficient = 0.60
        self.glazing_shgc = 0.67  # anywhere from 0.25 to 0.9

        if RANDOMIZE_HOUSE:
            self.glazing_shgc = random.gauss(0.67, 0.23)

        self.Rroof = 30.0
        self.Rwall = 19.0
        self.Rfloor = 22.0
        self.Rwindows = 1 / 0.47  # anywhere from 1/0.34 to 1/1.27

        self.Rdoors = 5.0

        if RANDOMIZE_HOUSE:
            self.Rwindows = random.gauss(1 / 0.6, 1 / 5)
            self.Rdoors = 5.0 + random.uniform(-1, 1)

        self.air_density = 0.0735  # density of air [lb/cf]
        self.air_heat_capacity = 0.2402  # heat capacity of air @ 80F [BTU/lb/F]
        self.volume = self.ceiling_height * self.floor_area  # volume of air [cf]
        self.air_mass = self.air_density * self.volume  # mass of air [lb]
        self.air_thermal_mass = 3 * self.air_heat_capacity * self.air_mass  # thermal mass of air [BTU/F]  //*3 multiplier is to reflect that the air mass includes surface effects from the mass as well.
        self.mass_solar_gain_fraction = 0.5  # Rob Pratt's implimentation for heat gain from the solar gains
        self.mass_internal_gain_fraction = 0.5  # Rob Pratt's implimentation for heat gain from internal gains
        self.total_thermal_mass_per_floor_area = 2.0
        self.airchange_per_hour = 0.5

        if RANDOMIZE_HOUSE:
            self.total_thermal_mass_per_floor_area = 2.0 + random.uniform(-0.3, 0.4)
            self.airchange_per_hour = 0.5 + random.uniform(-0.1, 0.3)

        self.interior_surface_heat_transfer_coeff = 1.46
        self.airchange_UA = self.airchange_per_hour * self.volume * self.air_density * self.air_heat_capacity
        self.door_area = self.number_of_doors * 3.0 * 78.0 / 12.0
        self.window_area = self.gross_wall_area * self.window_wall_ratio * self.exterior_wall_fraction
        self.net_exterior_wall_area = self.exterior_wall_fraction * self.gross_wall_area - self.window_area - self.door_area
        self.exterior_ceiling_area = self.floor_area * self.exterior_ceiling_fraction / self.number_of_stories
        self.exterior_floor_area = self.floor_area * self.exterior_floor_fraction / self.number_of_stories
        self.envelope_UA = self.exterior_ceiling_area / self.Rroof + self.exterior_floor_area / self.Rfloor \
                           + self.net_exterior_wall_area / self.Rwall + self.window_area / self.Rwindows + self.door_area / self.Rdoors
        self.UA = self.envelope_UA + self.airchange_UA
        self.solar_heatgain_factor = self.window_area * self.glazing_shgc * self.window_exterior_transmission_coefficient
        self.heating_setpoint = 70.0
        self.cooling_setpoint = random.randint(72, 76)
        self.cooling_setpoint = AC_BASE_SETPOINT
        self.after_event_setpoint = AC_BASE_SETPOINT
        if RANDOMIZE_SETPOINTS:
            self.cooling_setpoint = AC_BASE_SETPOINT + random.randint(0, 8) / 2.0

        # self.cooling_setpoint = 74
        self.design_cooling_setpoint = 75.0
        self.design_heating_setpoint = 70.0
        self.design_peak_solar = 195.0
        self.thermostat_deadband = 2.0
        self.thermostat_cycle_time = 120
        self.heating_supply_air_temp = 150
        self.cooling_supply_air_temp = 50

        self.Thigh = self.cooling_setpoint + self.thermostat_deadband / 2.0
        self.Tlow = self.heating_setpoint - self.thermostat_deadband / 2.0
        self.Thigh = clip(self.Thigh, 60.0, 140.0)
        self.Tlow = clip(self.Tlow, 60.0, 140.0)

        self.Tair = AC_BASE_SETPOINT  # random.uniform(self.Tlow,self.Thigh)

        if RANDOMIZE_HOUSE:
            # self.Tair = random.gauss(self.cooling_setpoint, 2)
            self.Tair = random.uniform(self.cooling_setpoint - 1, self.cooling_setpoint + 1)

        self.over_sizing_factor = 0.0
        self.cooling_design_temperature = 95.0
        self.heating_design_temperature = 70
        self.design_internal_gains = 167.09 * (self.floor_area ** 0.442)
        self.latent_load_fraction = 0.30  # effect of latent heat on cooling capacity
        self.design_cooling_capacity = (1.0 + self.over_sizing_factor) * (
                1.0 + self.latent_load_fraction) \
                                       * (self.UA * (
                self.cooling_design_temperature - self.design_cooling_setpoint)
                                          + self.design_internal_gains + (
                                                  self.design_peak_solar * self.solar_heatgain_factor))
        self.design_cooling_capacity = math.ceil(
            self.design_cooling_capacity / 6000) * 6000  # design_cooling_capacity is rounded up to the next 6000 btu/hr
        self.heating_system_type = 'heat_pump'
        if self.heating_system_type == 'heat_pump':
            self.design_heating_capacity = self.design_cooling_capacity  # /* primary is to reverse the heat pump */
        else:
            self.design_heating_capacity = (1.0 + self.over_sizing_factor) * self.UA * (
                    self.design_heating_setpoint - 32)
            self.design_heating_capacity = math.ceil(
                self.design_heating_capacity / 10000) * 10000  # design_heating_capacity is rounded up to the next 10,000 btu/hr

        ##For FAN ############
        self.duct_pressure_drop = 0.5
        self.aux_heat_capacity = (1.0 + self.over_sizing_factor) * self.UA * \
                                 (self.design_heating_setpoint - self.heating_design_temperature)
        self.aux_heat_capacity = math.ceil(self.aux_heat_capacity / 10000.0) * 10000.0

        if self.design_heating_capacity > self.aux_heat_capacity:
            self.design_heating_cfm = self.design_heating_capacity / (self.air_density * self.air_heat_capacity * (
                    self.heating_supply_air_temp - self.design_heating_setpoint)) / 60.0
        else:
            self.design_heating_cfm = self.aux_heat_capacity / (self.air_density * self.air_heat_capacity * (
                    self.heating_supply_air_temp - self.design_heating_setpoint)) / 60.0

        self.design_cooling_cfm = self.design_cooling_capacity / (1.0 + self.latent_load_fraction) / (
                self.air_density * self.air_heat_capacity * (
                self.design_cooling_setpoint - self.cooling_supply_air_temp)) / 60.0

        self.gtr_cfm = self.design_heating_cfm if self.design_heating_cfm > self.design_cooling_cfm else self.design_cooling_cfm
        self.fan_design_airflow = self.gtr_cfm

        self.fan_design_power = math.ceil(
            (0.117 * self.duct_pressure_drop * self.fan_design_airflow / 0.42 / 745.7) * 8) / 8.0 * 745.7 / 0.88
        #######################

        self.house_content_thermal_mass = self.total_thermal_mass_per_floor_area * self.floor_area - 2 * self.air_heat_capacity * self.air_mass
        self.house_content_heat_transfer_coeff = self.interior_surface_heat_transfer_coeff \
                                                 * ((
                                                            self.gross_wall_area - self.window_area - self.door_area)
                                                    + self.gross_wall_area
                                                    * self.interior_exterior_wall_ratio + self.number_of_stories
                                                    * self.exterior_ceiling_area / self.exterior_ceiling_fraction)
        # heat transfer coefficient of house_e contents [BTU/hr.F]

        self.Tmaterials = self.Tair
        self.hvac_motor_efficiency = 0.874  # anywhere from 0.82 (poor) to 0.92 (very good)
        self.system_rated_capacity = -self.design_cooling_capacity * 0.67  # TODO use COP curve from gridlabd to adjust according to temperature

        # if OPERATION_MODE == COOLING_MODE:
        #     self.system_rated_capacity =

        self.internal_gain = 3000  # Internal heat gain due to appliances
        self.window_open = False
        self.ua_factor = 1

        # a = house_content_thermal_mass*air_thermal_mass/house_content_heat_transfer_coeff
        self.set_point = random.randint(75, 79)
        self.temperature = self.set_point + random.randint(-1, 4)
        self.deadband = 1
        self.running = 0  # random.randint(0,1)
        self.set_status = 0
        # start with 0 power

        self.set_point_update = self.env.event()  # type: simpy.Event
        self.state_update_event = self.env.event()

        self.dr_event_running = False
        self.hvac_power = HVAC_POWER
        self.incident_solar_radiation = inputData[EDType.IncidentSolar][int(self.env.now / 3600)]
        if self.window_open:
            self.ua_factor = 10
        else:
            self.ua_factor = 1

        self.last_update_time = self.env.now
        self.thermal_parameters = self.updateModel()

        self.last_power = 0
        # self.env.process(self.updatePower(0))

        self.upper_comfort_boundary = self.cooling_setpoint + UPPER_BOUNDARY_OFFSET  # + random.randint(0,2)
        self.lower_comfort_boundary = self.cooling_setpoint - LOWER_BOUNDARY_OFFSET  # - random.randint(0,2)


    def __repr__(self):
        return f"{self.app_type} {self.app_id}, Temp: {self.Tair}, Status: {self.running}"

    def get_turn_on_priority_weight(self):
        #lower weight, higher the priority
        if self.running:
            model = self.updateModel(status=0)
        else:
            model = self.thermal_parameters
        time_to_boundary = self.next_hvac_event_time(82, model)
        if math.isnan(time_to_boundary):
            time_to_boundary = 0
        return time_to_boundary

    def can_turn_on(self):
        min_temp = 75 if self.inDR else 76
        return self.Tair > min_temp

    def can_turn_off(self):
        max_temp = 82 if self.inDR else 78
        return self.Tair < max_temp

    def must_turn_on(self):
        max_temp = 82 if self.inDR else 78
        return self.Tair > max_temp

    def must_turn_off(self):
        min_temp = 75 if self.inDR else 76
        return self.Tair < min_temp

    @property
    def rated_power(self):
        return self.thermal_parameters.hvac_power*1000

    def updateModel(self,Tout=None,incident_solar_radiation=None,status=None,Ta=None,Tm=None):
        #returns thermal parameters so that the temperature in the future can be predicted

        current_hour = int(self.env.now / 3600)
        w = float(self.env.now / 3600.0) - current_hour
        #w = 0  # force zero-order interpolation like in gridlabd

        #Tinternal = self.design_internal_gains #default is this, but will be updated

        Tinternal = inputData[EDType.HeatGain][self.env.now] #reading internal heat_gains from file. This should ideally
        #be computed based on internal appliance usages, and occupancy changes

        if Tout==None:
            Tout = inputData[EDType.OutdoorTemp][self.env.now]

        # if Tout > 75:
        #     print("Great")
        if incident_solar_radiation == None:
            incident_solar_radiation = inputData[EDType.IncidentSolar][self.env.now]

        if status == None:
            status = self.running

        #logic from gridlab-d for calculating the cooling_cop_adj
        if Tout < 40:
            temp_temperature = 40
        elif Tout > 80:
            temp_temperature = Tout
        else:
            temp_temperature = Tout

        cooling_cop_adj = self.cooling_cop / (-0.01363961 + 0.01066989*temp_temperature) # Line 01876 http://www.gridlabd.org/documents/doxygen/latest_dev/house__e_8cpp-source.html
        cooling_capacity_adj = self.design_cooling_capacity*(1.48924533 - 0.00514995*(Tout)) #Line 01892

        system_rated_capacity = -cooling_capacity_adj
        cooling_demand = cooling_capacity_adj/ cooling_cop_adj * KWPBTUPH

        #system_rated_capacity = -cooling_capacity_adj * voltage_adj + self.fan_power * BTUPHPKW * self.fan_heatgain_fraction

        hvac_power = cooling_capacity_adj/ cooling_cop_adj * KWPBTUPH

        Ca = self.air_thermal_mass
        Cm = self.house_content_thermal_mass
        Ua = self.UA
        Hm = self.house_content_heat_transfer_coeff
        # Qs = solar_load
        # Qh = load.heatgain
        if not Ta:
            Ta = self.Tair
        # dTa = dTair
        if not Tm:
            Tm = self.Tmaterials

        a = Cm * Ca / Hm
        b = Cm * (self.ua_factor * Ua + Hm) / Hm + Ca
        c = Ua * self.ua_factor
        c1 = -(Ua * self.ua_factor + Hm) / Ca
        c2 = Hm / Ca
        rr = math.sqrt(b * b - 4 * a * c) / (2 * a)
        r = -b / (2 * a)
        r1 = r + rr
        r2 = r - rr
        A3 = Ca / Hm * r1 + (self.ua_factor * Ua + Hm) / Hm
        A4 = Ca / Hm * r2 + (   self.ua_factor * Ua + Hm) / Hm

        Qs = incident_solar_radiation * self.solar_heatgain_factor
        if Qs>0:
            xxx=1


        Qi = Tinternal
        Qa = system_rated_capacity * status + (1 - self.mass_internal_gain_fraction) * Qi + (1 - self.mass_solar_gain_fraction) * Qs
        Qm = self.mass_internal_gain_fraction * Qi + self.mass_solar_gain_fraction * Qs
        d = Qa + Qm + (self.ua_factor * Ua) * Tout
        g = Qm / Hm

        current_minute = int(self.env.now/60) - current_hour*60
        current_second = self.env.now - (current_hour*3600+current_minute*60)

        Teq = d / c
        dTa = c2 * Tm + c1 * Ta - (c1 + c2) * Tout + Qa / Ca
        A1 = (r2 * Ta - r2 * Teq - dTa) / (r2 - r1)
        A2 = Ta - Teq - A1

        return THERMAL_PARAMETERS(A1, A2, A3, A4, r1, r2, a, b, c, d, g, hvac_power)

    def getTemps(self,thermal_parameters,dt):
        dt = dt/3600 #needs to be in hour
        tp = thermal_parameters
        e1 = tp.A1 * math.exp(tp.r1 * dt)
        e2 = tp.A2 * math.exp(tp.r2 * dt)
        Tair = e1 + e2 + tp.d / tp.c
        Tmaterials = tp.A3 * e1 + tp.A4 * e2 + tp.g + (tp.d / tp.c)
        return Tair, Tmaterials

    def next_hvac_event_time(self, Tevent: float, thermal_parameters: THERMAL_PARAMETERS = None):
        #called sync_house in old code

        tp = thermal_parameters or self.thermal_parameters #use the supplied tp or the house's built-in if tp is None
        dt2 = solver.e2solve(tp.A1, tp.r1, tp.A2, tp.r2, tp.d / tp.c - Tevent)
        return (dt2 * 3600)

    def next_event_time(self):
        if self.running == 1:
            Tevent = 75 if self.inDR else self.cooling_setpoint - self.thermostat_deadband/2
        else:
            Tevent = 82 if self.inDR else self.cooling_setpoint + self.thermostat_deadband/2
        return self.next_hvac_event_time(Tevent, self.thermal_parameters)


    @property
    def runCycles(self):
        Tevent = self.cooling_setpoint - self.thermostat_deadband/2
        run_time = self.next_hvac_event_time(Tevent,self.thermal_parameters)
        return [(run_time,self.thermal_parameters.hvac_power*1000)]


    @property
    def max_load_for_dr(self):
        return 5

    @property
    def load(self):
        return abs(self.Tair - self.cooling_setpoint)

    def applyControlLogic(self, external_state_request=None):

        if not self.inDR:
            if self.Tair > self.cooling_setpoint + self.thermostat_deadband/2 - EPS and self.running == 0:
                self.changeStatus(1) #the temperature too high; must turn on
                self.changePower(self.thermal_parameters.hvac_power*1000)
                self.thermal_parameters = self.updateModel()
            elif self.Tair < self.cooling_setpoint - self.thermostat_deadband/2 - EPS and self.running == 1:
                self.changeStatus(0) #temperature too low; must turn off
                self.changePower(0)
                self.thermal_parameters = self.updateModel()
        else:
            if external_state_request is not None:
                if external_state_request == 1 and self.running == 0:
                    self.changeStatus(1)  # the temperature too high; must turn on
                    self.changePower(self.thermal_parameters.hvac_power * 1000)
                    self.thermal_parameters = self.updateModel()
                elif external_state_request == 0 and self.running == 1:
                    self.changeStatus(0)  # the temperature too high; must turn on
                    self.changePower(0)
                    self.thermal_parameters = self.updateModel()

    def updateInternalStates(self, time=None):
        if not time:
            return
        self.Tair, self.Tmaterials = self.getTemps(self.thermal_parameters, dt=time)
        self.thermal_parameters = self.updateModel()
        self.temp_record.append((self.env.now, self.Tair))


class House:
    def __init__(self, house_id, dr_aggregator: DRAggregator, **kwargs):
        self.env = ENV
        self.dr_aggregator = dr_aggregator
        self.dr_aggregator.register(self)
        self.appliances: Dict[ApplianceType, GenericAppliance] = dict()
        self.appliances[ApplianceType.HVAC] = HVAC(house_id, ApplianceType.HVAC, self, hvac_aggregator, **kwargs)
        self.appliances[ApplianceType.WaterHeater] = WaterHeater(house_id, ApplianceType.WaterHeater, self, water_heater_aggregator)
        self.appliances[ApplianceType.Dishwasher] = Dishwasher(house_id, ApplianceType.Dishwasher, self, dishwasher_aggregator)
        self.appliances[ApplianceType.Washer] = Washer(house_id, ApplianceType.Washer, self, washer_aggregator)
        self.appliances[ApplianceType.Dryer] = Dryer(house_id, ApplianceType.Dryer, self, dryer_aggregator)


        self.power_record: List[Tuple[float,float]] = []
        self.total_exact_power = 0
        self.appliance_powers: Dict[GenericAppliance,float] = dict()

        for appliance in self.appliances.values():
            self.appliance_powers[appliance] = 0

    def changePower(self,appliance,value):
        try:
            self.total_exact_power -= self.appliance_powers[appliance] #remove the old power from the count
        except KeyError:
            print("Caught the MF")
        self.appliance_powers[appliance] = value #update the current power of appliance
        self.total_exact_power += value #add it to the total

        if self.power_record and self.power_record[-1][0] == self.env.now:
            self.power_record.pop() #if the last entry is also for the same time, remove it and update it

        self.power_record.append((self.env.now, self.total_exact_power))
        self.dr_aggregator.changePower(self,self.total_exact_power)


washer_aggregator = ApplianceDriver( app_type=ApplianceType.Washer, load_shape_data=inputData[EDType.ClothesWasherLoadshape])
water_heater_aggregator = ApplianceDriver(app_type=ApplianceType.WaterHeater, load_shape_data=inputData[EDType.WaterHeaterLoadshape])
dryer_aggregator = ApplianceDriver(app_type=ApplianceType.Dryer, load_shape_data=inputData[EDType.DryerLoadshape])
dishwasher_aggregator = ApplianceDriver(app_type=ApplianceType.Dishwasher, load_shape_data=inputData[EDType.DishwasherLoadshape])
hvac_aggregator = ApplianceDriver(app_type=ApplianceType.HVAC)



def simulationProgress():
    while True:
        print(f"Time (hour): {ENV.now/3600}\n")
        yield  ENV.timeout(3600)

ENV.process(simulationProgress())

def simulate(start_hr, end_hr):
    inputData = EDLoader(simulation_start_time_hr=start_hr)
    ENV.run(until=(end_hr-start_hr)*3600)


