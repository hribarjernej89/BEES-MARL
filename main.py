"""
MIT License

Copyright (c) 2024 Jernej Hribar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import time
from sensor import Sensor
from cloudlet import CloudLet
import simpy
import numpy as np
import pandas as pd

# Updating Strategy CloudLet can adopt to control sensors
STRATEGY_GREED = "greedy"  # Always both sensors transmit
STRATEGY_ALTER = "alter"  # We constantly alter between different possible settings
STRATEGY_PROBABILITY = "probability"  # Sensors have 50 % probability to decide to transmit the data
STRATEGY_LEARN = "deep_learning"  # The deep Q-learning approach
STRATEGY_MARL = 'marl'  # Independent Q-learning # STILL in the TO DO Phase...


# Simulation parameters
NUM_DAYS = 4  # Selecting the number of days we use the simulation
SIM_TIME = NUM_DAYS * 86400 + 1  # The number of steps for simulation, we do simulation in days
SIM_DATASET_START = 21600  # The time step of the simulation once we read the dataset, 6 in the morning

BAT_SIZE = 10000  # The size of battery in mAh on sensor

DAYS_FOR_COMPARISON = 1  # The number of days we use for the end results

ENERGY_HARVESTING_SOLAR_PANEL_VOLTAGE = 6.0  # Voltage has a direct impact collected energy, datasheet value
ENERGY_HARVESTING_EFFICIENCY = 0.8  # There are losses due to conversion circuit, factor is selected based on
NUM_SOLAR_PANELS = 4  # The number of solar panels each sensors has
TIME_STEP_SIZE = 1

RANDOM_ENERGY = False  # If True than the simulation changes the energy cost of transmission randomly at each step
RANDOM_BANDWIDTH = False  # If True than the simulation changes bandwidth randomly at each step

NRG_TO_TRANSMIT_A_BIT = [25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]  # Energy cost to transmit a bit in nJ
BANDWIDTH = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # Bandwidth in MBps

DEFAULT_ENERGY_COST = 85.0 * 1e-9  # The default energy cost in nJ
DEFAULT_BANDWIDTH = 50.0 * 10 ** 6  # The default bandwidth in MBps

NUMBER_OF_SENSORS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


if __name__ == '__main__':
    time_start = time.time()

    # Init the list, we use to track the overall performance of the system
    list_both_sensors_down = []
    list_overall_accuracy_mean = []
    list_overall_accuracy_avg = []
    list_nrg_regret = []
    list_overall_lat_avg = []

    for i in range(1):

        # The most important parameters
        strategy = STRATEGY_MARL  # Change the stragey here
        trans_energy = DEFAULT_ENERGY_COST  # The cost of transmitting a bit
        bandwidth = DEFAULT_BANDWIDTH   # The selected Bandwidth

        # Initialize the simulation
        env = simpy.Environment()  # Create simulation env

        # Initialize Sensors, lidars or cameras
        num_sensors = NUMBER_OF_SENSORS[i]
        print("We selected  strategy ", strategy, " and we need", trans_energy,
              " joules to transmit a bit. The bandwidth is", bandwidth, "Number of sensors", num_sensors)
        unique_ids = [x + 1 for x in range(num_sensors)]  # The id should match the number of sensors
        Sensors = []  # List containing of all sensors
        for j in range(num_sensors):
            sensor = Sensor(unique_ids[j], bat_size=BAT_SIZE, start_time_step=SIM_DATASET_START,
                            time_step_size=TIME_STEP_SIZE, nrg_to_transmit_a_bit=trans_energy,
                            randomise_nrg_cost=RANDOM_ENERGY)
            Sensors.append(sensor)
        # Initialize CloudLet (edge server)
        num_cloudlets = 1
        unique_id_cloudlets = [x for x in range(num_cloudlets)]
        Cloudlets = []
        for j in range(num_cloudlets):
            cloudlet = CloudLet(unique_id_cloudlets[j], updating_policy=strategy, sensors=Sensors,
                                time_step_size=TIME_STEP_SIZE, start_time_step=SIM_DATASET_START
                                , bandwidth=bandwidth, randomise_bandwidth=RANDOM_BANDWIDTH)
            Cloudlets.append(cloudlet)

        # Append processes to the simulation - ORDER IS VERY IMPORTANT
        for x in Cloudlets:
            env.process(x.select_strategy(env))
            env.process(x.train_rl_agent(env))
            env.process(x.save_trained_ann(env))

        for x in Sensors:
            env.process(x.changing_cost_of_transmission(env))
            env.process(x.energy_harvesting(env))  # In the next we collect the energy from a harvester
            env.process(x.basic_operation(env))  # In the first step the sensor will consume the energy
            env.process(x.collect_simulation_information(env))

        for x in Cloudlets:
            env.process(x.process_frame(env))
            env.process(x.collect_simulation_information(env))
            env.process(x.changing_bandwidth(env))

        # Start simulation
        env.run(np.round(SIM_TIME / TIME_STEP_SIZE))

        # Print out performance results - for each sensor
        for sensor in Sensors:
            print("Camera with id:", sensor.unique_id, "has collected:", sensor.daily_nrg_available_for_collection,
                  "the maximal capacity is:", sensor.max_stor_cap_joul, "end level is:", sensor.daily_nrg_joul,
                  "end read line is:", sensor.daily_read_line, "end energy level is:", sensor.daily_nrg_level,
                  "energy wasted:", sensor.daily_nrg_regret,
                  "Camera is NOT working for this many time steps:", sensor.daily_down_time,
                  )
            # Make a dataframe to store the information from the sensor
            df_sensor = pd.DataFrame()
            df_sensor['available_energy'] = sensor.daily_nrg_available_for_collection
            df_sensor['end_of_day_energy_level'] = sensor.daily_nrg_level
            df_sensor['wasted_energy'] = sensor.daily_nrg_regret
            df_sensor['down_time'] = sensor.daily_down_time

            df_sensor.to_csv('results/' + strategy + str(num_sensors) + 'sensors' + str(sensor.unique_id) + '.csv')

        # Print cloudlet results for the
        print("Both sensors were in stand-by for this amount of steps:", Cloudlets[0].daily_all_sens_down,
              "Daily mean accuracy in stand-by for this amount of steps:", Cloudlets[0].daily_mean_acc,
              "Daily average accuracy in stand-by for this amount of steps:", Cloudlets[0].daily_avg_acc
              )
        print("The selected accuracy was:", np.mean(Cloudlets[0].acc_save))  # This is the last day value
        # print("We missed the ground truth for this much:", np.mean(Cloudlets[0].grd_trh_diff))

        df_cloudlet = pd.DataFrame()
        df_cloudlet['both_sensors_down'] = Cloudlets[0].daily_all_sens_down
        df_cloudlet['accuracy_avg'] = Cloudlets[0].daily_mean_acc
        df_cloudlet['accuracy_mean'] = Cloudlets[0].daily_avg_acc
        df_cloudlet['average_lat'] = Cloudlets[0].daily_avg_lat

        df_cloudlet.to_csv('results/' + strategy + '_cloudlet_for_' + str(num_sensors) + '_sensors' + '.csv')

        # Append end results to the list - we only care about last 7 days
        list_both_sensors_down.append(np.sum(Cloudlets[0].daily_all_sens_down[-DAYS_FOR_COMPARISON:]))
        list_overall_accuracy_mean.append(np.average(Cloudlets[0].daily_mean_acc[-DAYS_FOR_COMPARISON:]))
        list_overall_accuracy_avg.append(np.average(Cloudlets[0].daily_avg_acc[-DAYS_FOR_COMPARISON:]))
        list_overall_lat_avg.append(np.average(Cloudlets[0].daily_avg_lat[-DAYS_FOR_COMPARISON:]))

        # Write end results for a specific updating strategy -
    df_results = pd.DataFrame()  # Dataframe
    df_results['Num_sensors'] = NUMBER_OF_SENSORS
    df_results['acc_mean'] = list_overall_accuracy_mean
    df_results['acc_avg'] = list_overall_accuracy_avg
    df_results['down_time'] = list_both_sensors_down
    df_results['latency'] = list_overall_lat_avg

    df_results.to_csv('results/' + strategy + '.csv')

    time_end = time.time()
    time_total = time_end - time_start
    print("Simulation Ended and lasted for ", time_total, " s")

