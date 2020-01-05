import re
import numpy as np

def read_log_dyn_press(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" AIRSPEED_MS45XX (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return np.array(list_meas)

def read_log_SDP3X(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" AIRSPEED_SDP3X (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return np.array(list_meas)

def read_log_aoa_flags(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" AOA (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return np.array(list_meas)
    

def read_log_aoa_press(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" PAYLOAD_FLOAT (\S+),(\S+),(\S+),(\S+),(\S+),(\S+),(\S+),(\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)), float(m.group(8)), float(m.group(9))])
    return np.array(list_meas)

def read_log_payload4(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" PAYLOAD_FLOAT (\S+),(\S+),(\S+),(\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))])
    return np.array(list_meas)
    
def read_log_gps(ac_id, filename):
    """Extracts gps values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" GPS (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), 
           float(m.group(7)), float(m.group(8)), float(m.group(9)), float(m.group(10)), float(m.group(11)),float(m.group(12))])
    return np.array(list_meas)
    
    
def read_log_attitude(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" ATTITUDE (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return np.array(list_meas)
    
def read_log_actuators(ac_id, filename):
    """Extracts ACTUATOR values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" ACTUATORS (\S+),(\S+),(\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return np.array(list_meas)
    
def read_log_energy(ac_id, filename):
    """Extracts Energy sensor values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" ENERGY (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))])
    return np.array(list_meas)

def read_log_energy_new(ac_id, filename):
    """Extracts New Energy sensor values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" ENERGY (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))])
    return np.array(list_meas)

def read_log_air_data(ac_id, filename):
    """Extracts Air-data values from a log.  Ps, Pd, temp,qnh, amsl_baro, airspeed, TAS"""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" AIR_DATA (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)), float(m.group(8))])
    return np.array(list_meas)

# wx wz Va gamma AoA Theta_commanded

def read_log_gust(ac_id, filename):
    """Extracts Energy sensor values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" GUST (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))])
    return np.array(list_meas) 

def read_log_imuaccel(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" IMU_ACCEL (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return np.array(list_meas)

def read_log_imugyro(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" IMU_GYRO (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return np.array(list_meas)

def read_log_mode(ac_id, filename):
    """Extracts mode values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" PPRZ_MODE (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))])
    return np.array(list_meas)

def read_log_settings(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" SETTINGS (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)),  float(m.group(5))])
    return np.array(list_meas)