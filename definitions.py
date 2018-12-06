from enum import Enum, unique
from collections import namedtuple
class RunType(Enum):
    Off = 1
    PartialRun = 2
    CompleteRun = 3


EDOptions = namedtuple("EDFormat", "filename, factor, period, offset_hr, should_interpolate")

@unique
class EDType(Enum):
    #'fileName', multiplier-factor, time_period, offset_hr, interpolate (1 or 0)
    HeatGain = EDOptions('heat_gains.txt', 1, 3600, 0, True)
    OutdoorTemp = EDOptions('outdoor_temperature.txt', 1, 3600, 0, True)
    MarketPrice = EDOptions('second_prices.txt', 1, 300, 0, False)
    IncidentSolar = EDOptions('va_incident_solar.txt', 1, 3600, 0, True)
    EndUse = EDOptions('end_use.txt', 1, 3600, 0, True)
    WaterHeaterLoadshape = EDOptions('waterheateravgkw.txt', 1000, 3600, 0, False)
    ClothesWasherLoadshape = EDOptions('clotheswasheravgkw.txt', 1000, 3600, 0, False)
    DryerLoadshape = EDOptions('dryeravgkw.txt', 1000, 3600, 0, False)
    DishwasherLoadshape = EDOptions('dishwasheravgkw.txt', 1000, 3600, 0, False)
    WaterUsageFraction = EDOptions('water_usage_fraction.txt', 1, 3600, 0, False)

#RegData = EDOptions('RegDComplete.txt',1,2,0,False)

#All data may not be available starting January 1. Offset determines the offset since beginning of the year. If the data
# begins on January 10th, the offset_hr would be 9 (10-1).

#whether to interpolate the data or just use zero order old

@unique
class ApplianceType(Enum):
    HVAC = "HVAC"
    WaterHeater = "WaterHeater"
    Dishwasher = "Dishwasher"
    Dryer = "Dryer"
    Washer = "Washer"

@unique
class DRType(Enum):
    LoadReduction =  1
    Regulation = 2

ApplianceLoadShapes = {
    ApplianceType.HVAC: None,
    ApplianceType.WaterHeater: EDType.WaterHeaterLoadshape,
    ApplianceType.Dishwasher: EDType.DishwasherLoadshape,
    ApplianceType.Dryer: EDType.DryerLoadshape,
    ApplianceType.Washer: EDType.ClothesWasherLoadshape
}

