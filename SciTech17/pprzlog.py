import re
import numpy as np

def read_log_adc_generic(ac_id, filename):
    """Extracts generic adc values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" ADC_GENERIC (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return np.array(list_meas)

def read_log_payload(ac_id, filename):
    """Extracts gps values from a log."""
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

   # <message name="GPS" id="8">
   #   <field name="mode"       type="uint8"  unit="byte_mask"/>
   #   <field name="utm_east"   type="int32"  unit="cm" alt_unit="m"/>
   #   <field name="utm_north"  type="int32"  unit="cm" alt_unit="m"/>
   #   <field name="course"     type="int16"  unit="decideg" alt_unit="deg"/>
   #   <field name="alt"        type="int32"  unit="mm" alt_unit="m">Altitude above geoid (MSL)</field>
   #   <field name="speed"      type="uint16" unit="cm/s" alt_unit="m/s">norm of 2d ground speed in cm/s</field>
   #   <field name="climb"      type="int16"  unit="cm/s" alt_unit="m/s"/>
   #   <field name="week"       type="uint16" unit="weeks"/>
   #   <field name="itow"       type="uint32" unit="ms"/>
   #   <field name="utm_zone"   type="uint8"/>
   #   <field name="gps_nb_err" type="uint8"/>
   # </message>
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


def read_log_esc(ac_id, filename):
    """Extracts esc  values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" ESC (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), 
           float(m.group(7)), float(m.group(8))])
    return np.array(list_meas)


  # <message name="AIR_DATA" id="222">
  #   <field name="pressure" type="float" unit="Pa">static pressure</field>
  #   <field name="diff_p" type="float" unit="Pa">differential pressure</field>
  #   <field name="temp" type="float" unit="deg celcius">air temperature</field>
  #   <field name="qnh" type="float" unit="hPa">barometric pressure adjusted to sea level</field>
  #   <field name="amsl_baro" type="float" unit="m">barometric altitude above mean sea level</field>
  #   <field name="airspeed" type="float" unit="m/s">Equivalent Air Speed (or Calibrated Air Speed at low speed/altitude)</field>
  #   <field name="tas" type="float">True Air Speed (when P, T and P_diff are available)</field>
  # </message>
def read_log_airdata(ac_id, filename):
    """Extracts esc  values from a log."""
    f = open(filename, 'r')
    pattern = re.compile("(\S+) "+ac_id+" AIR_DATA (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)")
    list_meas = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        m = re.match(pattern, line)
        if m:
           list_meas.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), 
           float(m.group(7)), float(m.group(8))])
    return np.array(list_meas)

  # <message name="IMU_ACCEL_SCALED" id="132">
  #   <field name="ax"    type="int32" alt_unit="m/s2" alt_unit_coef="0.0009766"/>
  #   <field name="ay"    type="int32" alt_unit="m/s2" alt_unit_coef="0.0009766"/>
  #   <field name="az"    type="int32" alt_unit="m/s2" alt_unit_coef="0.0009766"/>
  # </message>
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

#EOF