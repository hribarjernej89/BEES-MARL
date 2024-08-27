import numpy as np
import os
import ast

ENERGY_HARVESTER_DATASET_LENGTH = 1380400  # We have 16 days of energy-harvesting data

# Modes of operation for sensor
MODE_TRAN = 0  # In this mode sensors transmits captured
MODE_ANALYSE = 1  # In this mode camera does object detection with tiny Yolo, and transmit results
MODE_STAND_BY = 2  # Sensor does nothing
MODE_PWR_OFF = 3  # Sensor has no more energy

# Energy harvesting options, randomly assigned to the sensor
EH_DATASET_OPTIONS = ['13_3', '13_4']  # List of dataset we can use to model the energy arrival
EH_DATASET_LENGTH = 2  # The number of elements in the EH list
GEN_SIM_FPS = 10  # Decide on the number for the simulation
FRAME_SIZE = 500000  # The reserve in energy the sensor needs to have to be considered still active

# Energy parameters, the minimal and maximal values to transmit a bit
COST_OF_TRANS_BIT_MIN = 25  # The minimal cost of transmitting a bit in nJ
COST_OF_TRANS_BIT_MAX = 85  # The maximal cost of transmitting a bit in nJ


class Sensor:
    """
    A general sensor, that can only decide to transmit the acquired information. In general, we assume that the energy
    consumes is  on pair with a raspbery pi. However, they could be lower as lidar's total power is around 8W.
    """

    def __init__(self, unique_id, num_of_solar_panels=4, time_step_size=1, start_time_step=0,
                 bat_size=12000, battery_voltage=5.15, start_nrg_level=0.1, nrg_to_transmit_a_bit=46.50 * 1e-9,
                 basic_op_voltage=5.15, basic_op_current=0.270, solar_panel_voltage=6.1,
                 solar_panel_efficiency=0.8,  start_mode_operation=MODE_TRAN,
                 randomise_nrg_cost=False):

        self.unique_id = unique_id

        self.start_time_step = start_time_step  # To set when in the day was camera deployed
        self.time_step_size = time_step_size  # The length of the decision epoch in seconds
        self.time_step = self.start_time_step

        # Energy related Information
        self.battery_size = bat_size  # Battery size in mAh
        self.battery_voltage = battery_voltage
        self.max_stor_cap_joul = self.battery_size * self.battery_voltage * 3.6  # Calculating avail. energy
        self.start_nrg_level = start_nrg_level  # Starting energy in percentage
        self.nrg_joul = self.max_stor_cap_joul * self.start_nrg_level  # Energy in Joules
        self.nrg_level = np.round(self.nrg_joul / self.max_stor_cap_joul, 3) * 100 # Available energy in percentage

        # The Basic operation consumption
        self.basic_op_voltage = basic_op_voltage
        self.basic_op_current = basic_op_current
        self.nrg_consumption_step = self.basic_op_voltage*self.basic_op_current*self.time_step_size

        # Solar panel parameters
        self.num_of_solar_panels = num_of_solar_panels  # Number of panels control how much energy is collected
        self.solar_panel_voltage = solar_panel_voltage
        self.solar_panel_efficiency = solar_panel_efficiency
        # Reading from a file, we select either one or two, depending on the assigned ID
        self.harvesting_dataset_selection = EH_DATASET_OPTIONS[self.unique_id % EH_DATASET_LENGTH]

        # Reading real value from a file, save it to the
        self.nrg_har_values_file = open('energy_harvester/device' + self.harvesting_dataset_selection + '.csv', 'r')
        self.nrg_har_values = self.nrg_har_values_file.readlines()
        self.read_line = int(self.time_step * self.time_step_size)  # We read time
        self.nrg_har_real_values = self.nrg_har_values[self.read_line]
        # Reading measurements from the file
        self.nrg_har_PV_current = float(self.nrg_har_real_values.split(',')[0]) * 0.001  # mA
        self.nrg_har_wind_speed = float(self.nrg_har_real_values.split(',')[1])
        self.nrg_har_bat_current = float(self.nrg_har_real_values.split(',')[2])
        self.nrg_har_bat_voltage = float(self.nrg_har_real_values.split(',')[3])
        self.nrg_har_temperature = float(self.nrg_har_real_values.split(',')[4])
        self.nrg_har_humidity = float(self.nrg_har_real_values.split(',')[5])
        # Calculating arrived energy
        self.arrived_nrg = np.round(self.nrg_har_PV_current*self.solar_panel_voltage*self.time_step_size*
                                       self.num_of_solar_panels, 7)

        # Mode of operation
        self.mode_op = start_mode_operation  # We start in the mode in which we always transmit images
        self.mode_op_nrg_con = 0.0  # We init it as zero but it is determined in the first step

        # Energy consumption, how much the camera needs to transmit
        self.nrg_to_transmit_a_bit = nrg_to_transmit_a_bit   # 46.50 * 1e-9  # Energy needed to transmit a bit
        self.nrg_to_transmit_a_byte = 8 * self.nrg_to_transmit_a_bit
        self.min_nrg_requirement = FRAME_SIZE * int(GEN_SIM_FPS * self.time_step_size) * self.nrg_to_transmit_a_byte

        # Images it has to transmit
        self.img_to_process = [0]  # List of images the camera has to transmit/process
        self.image_sizes = []  # List of sizes, of the images we want to transmit

        # The idea is to see how much energy camera does not collect because of full energy
        self.nrg_available_for_collection = 0.0  # All available energy for collection
        self.nrg_regret = 0.0  # The energy the sensor can not collect due to a full battery

        # To measure how long is each camera not working even when it should:
        self.down_time = 0


        # Tracking daily performance of camera in terms of energy consumption
        self.daily_nrg_available_for_collection = []  # Energy collected in the day
        self.daily_nrg_joul = []  # Energy level after the day of
        self.daily_read_line = []  # Which line we are reading
        self.daily_nrg_level = []   # Energy level at end of the day
        self.daily_nrg_regret = []   # The regret in terms of energy we could have collected but we did not
        self.daily_down_time = []  # The amount of time cameras spent in down-time

        # Randomising the energy of transmission
        self.randomise_nrg_cost = randomise_nrg_cost

    def energy_harvesting(self, env):
        """
        This method collects energy based on the values the real values captured from the Solar dataset in paper:

        This method operates independently of camera capture part. We also assume that it works even if the camera's
        battery is fully depleted.
        """

        while True:
            yield env.timeout(1)
            # print("I the Fifth process!", env.now)
            self.time_step = self.time_step + 1
            self.read_line = int(self.time_step * self.time_step_size) % ENERGY_HARVESTER_DATASET_LENGTH
            # Read from file, the new values depending on the step
            self.nrg_har_real_values = self.nrg_har_values[self.read_line]
            self.nrg_har_PV_current = float(self.nrg_har_real_values.split(',')[0]) * 0.001  # mA
            self.nrg_har_wind_speed = float(self.nrg_har_real_values.split(',')[1])
            self.nrg_har_bat_current = float(self.nrg_har_real_values.split(',')[2])
            self.nrg_har_bat_voltage = float(self.nrg_har_real_values.split(',')[3])
            self.nrg_har_temperature = float(self.nrg_har_real_values.split(',')[4])
            self.nrg_har_humidity = float(self.nrg_har_real_values.split(',')[5])
            self.arrived_nrg = np.round(self.nrg_har_PV_current * self.solar_panel_voltage * self.time_step_size *
                                           self.num_of_solar_panels, 7)
            if self.arrived_nrg > 0.0:
                self.nrg_available_for_collection = self.nrg_available_for_collection + self.arrived_nrg
                if (self.nrg_joul + self.arrived_nrg) > self.max_stor_cap_joul:  # We gonna waste some energy
                    self.nrg_regret = self.nrg_regret + (self.nrg_joul + self.arrived_nrg - self.max_stor_cap_joul)
                    # print("We are wasting energy:", (self.nrg_joul + self.arrived_nrg - self.max_stor_cap_joul))

                self.nrg_joul = min(self.nrg_joul + self.arrived_nrg, self.max_stor_cap_joul)
            # Re-calculate the energy level in percentage
            self.nrg_level = np.round(self.nrg_joul / self.max_stor_cap_joul, 3) * 100

            # print("Energy we have now is:", self.energy_joul)

    def basic_operation(self, env):
        """
        This function does all the functionality all embedded cameras have to do to main its operation. Mostly, the
        energy consumption to operate as normal
        """

        while True:
            yield env.timeout(1)
            # print("I the Furth process!", env.now)
            self.nrg_consumption_step = self.basic_op_voltage*self.basic_op_current*self.time_step_size
            #print("Step consumption", self.nrg_consumption_step)
            self.nrg_joul = max(np.round(self.nrg_joul - self.nrg_consumption_step, 8), 0)
            #print("Predicted nrg at time:", env.now, "is:", self.predicted_nrg)
            if self.min_nrg_requirement > self.nrg_joul:
                self.mode_op = MODE_PWR_OFF  # Camera goes into stand by mode when it does not
                self.down_time = self.down_time + 1  # To track how long is the camera in the down_time mode

    def collect_simulation_information(self, env):
        """
        The information we collect to measure the performance of our simulation
        """
        while True:
            yield env.timeout(int(86400 / self.time_step_size))  # We take the daily performance once per day
            print("We are collecting daily information at time step:", env.now)
            self.daily_nrg_available_for_collection.append(self.nrg_available_for_collection)
            self.daily_nrg_joul.append(self.nrg_joul)
            self.daily_read_line.append(self.read_line)
            self.daily_nrg_level.append(self.nrg_level)
            self.daily_nrg_regret.append(self.nrg_regret)
            self.daily_down_time.append(self.down_time)

            # Reset the values back to their original, those that should be
            self.nrg_available_for_collection = 0
            self.down_time = 0
            self.nrg_regret = 0

    def changing_cost_of_transmission(self, env):
        """
        With this function we change the cost of transmitting a bit in the decision epoch
        """
        while True:
            yield env.timeout(1)  # We do this every decision epoch
            if self.randomise_nrg_cost:
                self.nrg_to_transmit_a_bit = np.round(np.random.uniform(COST_OF_TRANS_BIT_MIN, COST_OF_TRANS_BIT_MAX,
                                                                        1)[0], 2) * 1e-9
                self.nrg_to_transmit_a_byte = 8 * self.nrg_to_transmit_a_bit
                self.min_nrg_requirement = FRAME_SIZE * int(
                    GEN_SIM_FPS * self.time_step_size) * self.nrg_to_transmit_a_byte


