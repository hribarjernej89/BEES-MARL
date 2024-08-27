import numpy as np
import random
import collections
import ast  # To read accuracy files
import itertools
from dqn_agent import DQNAgent

# Updating Strategy CloudLet can adopt to control cameras
STRATEGY_GREED = "greedy"  # Always both cameras transmit
STRATEGY_ALTER = "alter"  # We constantly alter between different possible settings
STRATEGY_PROBABILITY = "probability"  # Sensors have 50 % probability to decide to transmit the data
STRATEGY_LEARN = "deep_learning"  # The deep Q-learning approach
STRATEGY_MARL = 'marl'  # Independent Q-learning # STILL in the TO DO Phase...
STRATEGY_NRG_SEL = 'energy_selection'  # Energy selection strategy

# Values for the selected strategy
NRG_SEL_PERCENTAGE = 0.75

# Lidar processing related values
LIDAR_MAX_NUM_ACT = 16  # The total number of actions the system with four lidar sensors can take
LIDAR_NUMB_SPL = 1980  # The number of samples we have collected, 3 minutes 18 seconds in total
LIDAR_SMPL_INT = 60  # The time interval we randomly change the pointer in the read data
LIDAR_MAX_OBJECTS_TO_DETECT = 5  # The maximal number of objects to detect
LIDAR_REAL_FPS = 10  # Get this number
LIDAR_SIM_FPS = 10  # We will use the same number in all our simulations
LIDAR_MAX_NUM_MODE = 2  # The total number of actions we use in our work
LIDAR_SEN1_FRAME_SIZE_BYTES = 480000  # On average size of the transmitted frame, we determined it as 300 000 points/s with 16B for each points, divide by 10 frames obtained per second
LIDAR_SEN_FRAME_SIZE_BYTES = 384000  # On average size of the transmitted frame, we determined it as 240 000 points/s with 16B for each points, divide by 10 frames obtained per second
LIDAR_FRAME_SIZE_SDV = 0.05  # We assume 5% percent vairance in the frame size

# General environment parameter settings
GEN_MEAN = [0.0, 0.17, 0.55, 0.75, 0.87, 0.93, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
GEN_SDV = [0.0, 0.277, 0.224, 0.182, 0.148, 0.121, 0.099, 0.082, 0.068, 0.057, 0.048, 0.041, 0.035, 0.03, 0.027, 0.024, 0.022, 0.02, 0.018, 0.017, 0.016]
GEN_MAX_NUM_OBJECTS = 17  # The maximal number of objects to detect
GEN_REAL_FPS = 10  # Get this number
GEN_SIM_FPS = 10  # Decide on the number for the simulation
GEN_MAX_NUM_MODE = 2  # The total number of actions we use in our work
GEN_FRAME_SIZE_BYTES = 384000  # On average size of the transmitted frame, we determined it as 240 000 points/s with 16B for each points, divide by 10 frames obtained per second
GEN_FRAME_SIZE_SDV = 0.05  # We assume 5% percent variance in the frame size

# Modes of operation for sensor
MODE_TRAN = 0  # In this mode sensors transmits captured
MODE_ANALYSE = 1  # In this mode camera does object detection with tiny Yolo, and transmit results
MODE_STAND_BY = 2  # Sensor does nothing
MODE_PWR_OFF = 3  # Sensor has no more energy

# DQN Agent related parameters
DQN_TRN_INT = 60  # We train the ANN every minute in the DQN agent every minute

# Latency related parameters - we keep the as we had i
LAT_CLOUDLET_PROC_TIME = 17.85 * .001  # xavier 30w latency value from early experience paper
LAT_VARY_CONSTANT = 0.1  # The processing latency on the device and the cloudlet varies ±10 %
BANDWIDTH_MAX = 50  # The minimal bandwidth value we allow in our work in MB
BANDWIDTH_MIN = 10  # The maximal bandwidth value we allow in our system in MB


class CloudLet:
    """
    CloudLet has a high processing power available to process images from cameras using YoLo V4 set, note that
    """

    def __init__(self, unique_id, updating_policy, sensors=[], time_step_size=1, start_time_step=0,
                 bandwidth=2*10**6, randomise_bandwidth=False, load_ann_name="trained",
                 save_ann_name="trained"
                 ):
        self.unique_id = unique_id
        # General simulation information
        self.time_step_size = time_step_size
        self.start_time_step = start_time_step  # Important to know where to start reading files
        self.time_step = self.start_time_step
        # Get the list of actions, based on the number of sensors under cloudlet control - permutations
        self.sensors = sensors  # List of sensors that transmit information, i.e., sensors under its control
        self.num_sensors = len(self.sensors)
        self.action_list = list(itertools.product([MODE_STAND_BY, MODE_TRAN], repeat=self.num_sensors))
        self.num_of_actions = len(self.action_list)
        print("The total number of actions is:", self.num_of_actions)
        # Loading and saving the ANN of the learned agents
        self.load_ann_name = load_ann_name
        self.save_ann_name = save_ann_name
        # Variables we need to determine the accuracy
        self.num_frames_to_process = int(GEN_SIM_FPS * self.time_step_size)
        self.trans_data = []  # List tracking the amount of data for each frame the sensor transmits
        self.transmit_mask = []  # Individual mask of masks
        self.default_transmit_mask = []   # The default mask we use to determine the final mask
        self.guard_transmit_mask = []  # The mask we use of the guard transmit
        self.list_of_transmit_mask = []  # Nested list of list to be used in my simulation
        self.temp_transmit_mask = []  # The list we use only temporarily
        self.temp_step_load = 0.0  # The variable that holds the load information
        self.step_accuracy = []  # The list we keep to determine the accuracy in the step
        self.acc_list = []  # To keep the track of overall accuracy
        self.guard_acc = 0.0
        self.list_of_latencies = [[] for x in range(self.num_sensors)]
        # Tracking the accuracy of selected updating strategy
        self.num_cams_obj = 0
        self.acc_save = []  # List of accuracy for each selected picture, i.e, is selected combination good enough
        self.num_obj_det_grd_trh = 0  # The ground truth...the real number of objects camera detected
        self.grd_trh = []  # An empty list that will contain the ground truth of number of objects detected
        # Determining the  guard interval
        self.guard = 0  # The id of the image we process to determine the reward and accuracy
        self.int_acc = []  # The accuracy during the interval
        self.acc_guard = 0  # The accuracy we can use as an input for the reward function
        self.num_obj_guard = 0  # number of objects in the guard interval
        self.guard_mode = []
        # Variables measuring performance of my system
        self.detected_objects = 0  # variable storing detection we carried out before
        self.all_sens_down_list_com = [MODE_PWR_OFF for x in range(self.num_sensors)]
        self.all_sens_down = 0  # Counting when/if all sensors are without any energy

        # Selecting the updating policy
        self.upd_pol = updating_policy
        # set guard interval to true if required:

        # Adding the latency to the system
        self.bandwidth = bandwidth  # The bandwidth is also a parameter with control during simulation
        self.randomise_bandwidth = randomise_bandwidth  # When True randomise bandwidth at each step of the simulation
        self.data_limit = self.bandwidth * self.time_step_size  # The amount of data we can actually transmit
        # The variables we use to track the latency
        self.avg_lat = 0.0
        self.bandwidth_limit = False  # When set to True, it means we are losing frames/images
        self.num_of_not_trans_img = 0  # The number of images that are not transmitted due to the bandwidth limit
        self.img_trans_lat = []  # The list of images we actually transmitted
        self.temp_data_load = 0.0  # The temporal data load variable we need to determine which images are transmitted
        self.combined_data_load = 0.0  # The data load in a given step

        if (self.upd_pol == STRATEGY_LEARN) or (self.upd_pol == STRATEGY_MARL):
            self.guard_mode_on = True
        else:
            self.guard_mode_on = False
        if self.upd_pol == STRATEGY_ALTER:
            print("The selected operating mode is Alternating")
            self.upd_str = self.action_list[0]  # We start with the first on the line...
            for i in range(self.num_sensors):
                self.sensors[i].mode_op = self.upd_str[i]
        elif self.upd_pol == STRATEGY_GREED:
            print("The selected operating mode is Greedy")
            for i in range(self.num_sensors):
                self.sensors[i].mode_op = MODE_TRAN
        elif self.upd_pol == STRATEGY_PROBABILITY:
            print("The selected operating mode is Probability")
            for i in range(self.num_sensors):
                if np.random.random(1)[0] > 0.5:
                    self.sensors[i].mode_op = MODE_TRAN
                else:
                    self.sensors[i].mode_op = MODE_STAND_BY
        elif self.upd_pol == STRATEGY_NRG_SEL:
            print("The selecting operating mode is Energy Selection")
            self.num_of_selected = int(np.round(self.num_sensors * NRG_SEL_PERCENTAGE))
            sen_nrg_levels = np.array([self.sensors[j].nrg_level for j in range(self.num_sensors)])
            sel_sensors = np.argpartition(sen_nrg_levels, -self.num_of_selected)[-self.num_of_selected:]
            for i in range(self.num_sensors):
                if i in sel_sensors:
                    self.sensors[i].mode_op = MODE_TRAN
                else:
                    self.sensors[i].mode_op = MODE_STAND_BY
        elif self.upd_pol == STRATEGY_LEARN:
            print("The selected operating mode is DQN Agent")
            # Init the  stuff we need for DQN Agent
            # Building learning agent....
            self.DQN_num_sts = 2 + self.num_sensors  # accuracy, num of objects, energy_level
            self.DQN_num_act = self.num_of_actions  #
            # Init Pytorch model
            self.DQN_agent = DQNAgent(self.DQN_num_sts, self.DQN_num_act)
            # Load the pre-trained network if possible...
            try:
                self.DQN_agent.load("trained_models/dqn_model_for_sen_" + self.ann_name_ext + ".h5")
            except:
                print("No pretrained model available!")

            # Build a state space
            self.DQN_state = np.array([self.guard_acc * 10, self.num_obj_guard] + [s.nrg_level / 10 for s in self.sensors])
            self.DQN_state = np.reshape(self.DQN_state, [1, self.DQN_num_sts])  # Put the state in the right format
            # Take an action
            self.act = self.DQN_agent.act(self.DQN_state)
            self.upd_str = self.action_list[self.act]  # Determine the new updating strategy
            # print("We selected this strategy in the start:", self.upd_str)
            # Push the strategy into the sensors
            for i in range(self.num_sensors):
                self.sensors[i].mode_op = self.upd_str[i]
            # Save the last state
            self.DQN_old_state = self.DQN_state
            self.DQN_rwd = 0  # The reward
        elif self.upd_pol == STRATEGY_MARL:
            print("The selected operating mode is BEES-MARL approach")
            # We need to create a DQN for each agent
            # Building learning agent
            self.marl_num_sts = 2 + self.num_sensors  # accuracy, num of objects, energy_level
            self.marl_num_act = GEN_MAX_NUM_MODE  # Number of modes for each sensor, 2 modes in total
            self.marl_max_num = np.floor(self.bandwidth/(GEN_FRAME_SIZE_BYTES * GEN_SIM_FPS))
            print("Maximal number to transmit in BEES-MARL system is:", self.marl_max_num)
            print("The input state space is:", self.marl_num_sts,
                  "The number of actions each agent can take:", self.marl_num_act)
            self.marl_rwd = 0  # Init the reward
            self.dqn_agents = []  # list of independent agents
            # Init N number of DQN agents...
            for i in range(self.num_sensors):
                agent = DQNAgent(self.marl_num_sts, self.marl_num_act)
                try:
                    agent.load("trained_models/marl_ann_sen_" + str(i) + "_" + self.save_ann_name + ".h5")
                except:
                    print("No pretrained model available!")

                self.dqn_agents.append(agent)  # append the agents
                # Set the energy consumption correctly on sensors...
                self.sensors[i].dqn_op_mode = True
            self.old_states = []
            self.old_actions = []
            for i in range(self.num_sensors):
                #  construct a state space from the perspective of the agent that just transmitted
                state_space = np.array([self.guard_acc * 10, self.num_obj_guard, self.sensors[i].nrg_level / 10] +
                                       [self.sensors[j].nrg_level / 10 for j in range(self.num_sensors) if j != i])
                state_space = np.reshape(state_space, [1, self.marl_num_sts])
                # Decide on the  action for i-th sensor
                act = self.dqn_agents[i].act(state_space)
                # save actions for the mememory
                self.old_actions.append(act)
                # Push the decision to the sensor
                self.sensors[i].mode_op = decode_action_into_strategy_marl(act)
                # Save the past state
                self.old_states.append(state_space)

        else:
            print("Select a valid updating strategy!!")

        # The simulation parameters we capture each day to measure the performance
        self.daily_mean_acc = []
        self.daily_avg_acc = []
        self.daily_all_sens_down = []
        self.daily_avg_lat = []

    def process_frame(self, env):
        """"
        In this process we read from files or using the extracted probability function to determine how accurate
        is a particular strategy. Additionally, we determine the latency which the selected mode
        incurs.
        """

        while True:
            yield env.timeout(1)
            # print("I trying to determine the accuracy!", env.now)
            # Generate the list of the amount of data to transmit
            self.trans_data = [np.random.normal(GEN_FRAME_SIZE_BYTES, GEN_FRAME_SIZE_SDV * GEN_FRAME_SIZE_BYTES,
                                                self.num_sensors) for i in range(self.num_frames_to_process)]
            # Step 1: Determine the data we need to transmit and collect default modes of operation
            self.default_transmit_mask = []
            for sensor in self.sensors:
                if sensor.mode_op == MODE_TRAN:
                    self.default_transmit_mask.append(1)
                else:
                    self.default_transmit_mask.append(0)
            self.list_of_transmit_mask = []
            # set the guard interval if necessary
            self.guard_transmit_mask = []
            self.temp_data_load = 0.0
            if self.guard_mode_on:
                for j in range(self.num_sensors):
                    if self.sensors[j] == MODE_PWR_OFF:
                        self.guard_transmit_mask.append(0)  # We can not transmit as there is no energy
                    elif (self.temp_data_load + self.trans_data[-1][j]) <= self.data_limit:
                        self.temp_data_load += self.trans_data[-1][j] * self.default_transmit_mask[j]
                        self.guard_transmit_mask.append(1)
                    else:
                        self.guard_transmit_mask.append(0)  # We can not transmit
            else:
                for j in range(self.num_sensors):
                    if self.temp_data_load + (
                            self.trans_data[-1][j] * self.default_transmit_mask[j]) <= self.data_limit:
                        self.temp_data_load += self.trans_data[-1][j] * self.default_transmit_mask[j]
                        self.guard_transmit_mask.append(self.default_transmit_mask[j])
                    else:
                        self.guard_transmit_mask.append(0)  # We can not transmit

            for i in range(self.num_frames_to_process - 1):
                if self.temp_data_load+np.sum(self.default_transmit_mask*self.trans_data[i]) <= self.data_limit:
                    self.temp_data_load += np.sum(self.default_transmit_mask*self.trans_data[i])
                    self.list_of_transmit_mask.append(self.default_transmit_mask)
                else:  # we can not transmit everything
                    self.temp_transmit_mask = []
                    for j in range(self.num_sensors):
                        if self.temp_data_load+(self.default_transmit_mask[j]*self.trans_data[i][j]) <= self.data_limit:
                            self.temp_data_load += self.default_transmit_mask[j]*self.trans_data[i][j]
                            self.temp_transmit_mask.append(self.default_transmit_mask[j])
                        else:
                            self.temp_transmit_mask.append(0)
                    self.list_of_transmit_mask.append(self.temp_transmit_mask)
            self.list_of_transmit_mask.append(self.guard_transmit_mask)
            # Step 2: Determine the latency based on the data we transmit
            # Step 4, determine the latency and energy consumption
            for l in range(self.num_sensors):
                sen_data_trans = np.sum([self.trans_data[k][l] * self.list_of_transmit_mask[k][l] for k in
                                        range(self.num_frames_to_process)])
                sen_num_frames = np.sum([self.list_of_transmit_mask[k][l] for k in
                                                                           range(self.num_frames_to_process)])
                self.list_of_latencies[l].append(determine_latency(self.bandwidth, sen_data_trans, sen_num_frames))
                # Determine the energy cost of the transmission
                self.sensors[l].nrg_joul = self.sensors[l].nrg_joul - sen_data_trans * self.sensors[l].nrg_to_transmit_a_byte
            # Step 5, determine accuracy based on the number fo sensors transmitting:
            self.step_accuracy = []
            for k in range(self.num_frames_to_process):
                self.step_accuracy.append(determine_gen_acc(np.sum(self.list_of_transmit_mask[k])))

            self.guard_acc = self.step_accuracy[-1]
            self.acc_list.append(np.average(self.step_accuracy))

            # TO DO: Step 7. check if all sensors are down
            # Make a list with all modes
            # Check if all of them are moder POWER DOWN
            all_modes = [sensor.mode_op for sensor in self.sensors]
            if all_modes == self.all_sens_down_list_com:
                self.all_sens_down = self.all_sens_down + 1

    def select_strategy(self, env):
        """
        This function selects the mode of operation for the cameras, we have a few options:
        THRESHOLD POLICY: Cameras updating strategy changes depending on the their energy levels
        ALTERNATING POLICY: Each decision epoch a different camera is transmitting images
        GREEDY: Sensors always transmit...
        LEARNED POLICY: Use my DQN model and information to make predictions
        BEES-MARL: Use the multi-agent approach
        """
        while True:
            yield env.timeout(1)
            if self.upd_pol == STRATEGY_ALTER:
                self.upd_str = self.action_list[env.now % self.num_of_actions]  # Determine the new updating strategy
                for i in range(self.num_sensors):
                    self.sensors[i].mode_op = self.upd_str[i]
            elif self.upd_pol == STRATEGY_GREED:
                for i in range(self.num_sensors):
                    self.sensors[i].mode_op = MODE_TRAN
            elif self.upd_pol == STRATEGY_PROBABILITY:
                for i in range(self.num_sensors):
                    if np.random.random(1)[0] > 0.5:
                        self.sensors[i].mode_op = MODE_TRAN
                    else:
                        self.sensors[i].mode_op = MODE_STAND_BY
            elif self.upd_pol == STRATEGY_NRG_SEL:
                sen_nrg_levels = np.array([self.sensors[j].nrg_level for j in range(self.num_sensors)])
                sel_sensors = np.argpartition(sen_nrg_levels, -self.num_of_selected)[-self.num_of_selected:]
                for i in range(self.num_sensors):
                    if i in sel_sensors:
                        self.sensors[i].mode_op = MODE_TRAN
                    else:
                        self.sensors[i].mode_op = MODE_STAND_BY
            elif self.upd_pol == STRATEGY_LEARN:
                self.take_dqn_action()  # Take the action as defined below
                # print("Selected actions is", [s.mode_op for s in self.sensors])
            elif self.upd_pol == STRATEGY_MARL:
                self.take_marl_action()  # Take the action as MARL

    def train_rl_agent(self, env):
        """
        We do not train the ANN at every time step, but periodically to limit spent computational power!
        """
        while True:
            yield env.timeout(int(DQN_TRN_INT/self.time_step_size))
            if self.upd_pol == STRATEGY_LEARN:
                self.DQN_agent.replay()
            elif self.upd_pol == STRATEGY_MARL:
                for i in range(self.num_sensors):
                    self.dqn_agents[i].replay()

    def save_trained_ann(self, env):
        """
        A function that saves a trained ANN, if we train the network.
        """
        while True:
            yield env.timeout(86400)  # Save ANN at the end of each day
            # self.batch_size = 6400
            if self.upd_pol == STRATEGY_LEARN:
                print("The exploration rate is:", self.DQN_agent.epsilon)
                self.DQN_agent.save("trained_models/dqn_model_for_sen_" + self.ann_name_ext + ".h5")
            elif self.upd_pol == STRATEGY_MARL:
                print("We are saving the ANN for MARL approach at", env.now)
                for i in range(self.num_sensors):
                    self.dqn_agents[i].save("trained_models/marl_ann_sen_" + str(i) + "_" + self.save_ann_name + ".h5")
                    print("The new exploration rate is:", self.dqn_agents[i].epsilon)

    def collect_simulation_information(self, env):
        """
        The information we collect to measure the performance of our simulation
        """
        while True:
            yield env.timeout(int(86400 / self.time_step_size))  # We take the daily performance once per day

            self.daily_mean_acc.append(np.mean(self.acc_list))
            self.daily_avg_acc.append(np.average(self.acc_list))
            self.daily_all_sens_down.append(self.all_sens_down)
            self.daily_avg_lat.append(np.average([np.average(x) for x in self.list_of_latencies]))
            # Reset the values back to their original, those that should be
            self.all_sens_down = 0  # Counting when/if both cameras are without any energy
            self.acc_list = []  #
            self.list_of_latencies = [[] for x in range(self.num_sensors)]

    def changing_bandwidth(self, env):
        """
        With this function we change the bandwidth at each step
        """
        while True:
            yield env.timeout(1)  # We do this every decision epoch
            if self.randomise_bandwidth:
                self.bandwidth = np.round(np.random.uniform(BANDWIDTH_MIN, BANDWIDTH_MAX, 1)[0], 2) * 10 ** 6

    def take_dqn_action(self):
        """
        The function to take the DQN action, i.e., we have a lot of actions
        """
        self.DQN_state = np.array([self.guard_acc * 10, self.num_obj_guard] + [s.nrg_level for s in self.sensors])
        self.DQN_state = np.reshape(self.DQN_state, [1, self.DQN_num_sts])  # Put the state in the right format
        # Determine the reward and save the experience
        self.DQN_rwd = determine_journal_rwd_signal(self.guard_acc)
        # print("The past action was:", self.act, "the rewards was:", self.DQN_rwd)
        self.DQN_agent.remember(self.DQN_old_state, self.act, self.DQN_rwd, self.DQN_state)
        # Take a new action
        self.act = self.DQN_agent.act(self.DQN_state)
        # print("The action we decided to take is:", self.act)
        self.upd_str = self.action_list[self.act]  # Determine the new updating strategy
        # Push the strategy into the cameras
        for i in range(self.num_sensors):
            self.sensors[i].mode_op = self.upd_str[i]
        # Save the last state
        self.DQN_old_state = self.DQN_state

    def take_marl_action(self):
        """
        The function to take the MARL based action, a way to limit the actions taken
        """
        # First we determine the new reward
        # Determine the reward and save the experience
        self.marl_rwd = determine_journal_rwd_signal(self.guard_acc)
        for i in range(self.num_sensors):
            #  construct a state space from the perspective of the agent that just transmitted
            state_space = np.array([self.guard_acc * 10, self.num_obj_guard, self.sensors[i].nrg_level / 10] +
                                   [self.sensors[j].nrg_level / 10 for j in range(self.num_sensors) if j != i])
            state_space = np.reshape(state_space, [1, self.marl_num_sts])
            # Save experiende to the right DQN agent
            self.dqn_agents[i].remember(self.old_states[i], self.old_actions[i], self.marl_rwd, state_space)

            # Decide on the  action for i-th sensor
            act = self.dqn_agents[i].act(state_space)
            self.old_actions[i] = act
            # Push the decision to the sensor
            self.sensors[i].mode_op = decode_action_into_strategy_marl(act)
            # Save the past state
            self.old_states[i] = state_space


"""
Helper functions for the CloudLet 
"""


def determine_journal_rwd_signal(acc):
    """
    Input: recall value - achieved accuracy. Note that the value is limited to an interval 0 and 1.
    The suggested reward function that naturally arises from the proposed system model. However, to speed up the
    learning we alter it slightly and limit it to an interval between -1 and 1.

    Return the reward signal based solely on the achieved recall value.
    """
    return (acc - .5) * 2.0


def restrict(val, minval, maxval):
    """
    To ensure that the number of generated objects is realistic, i.e., between 0 and maz number possible.
    """
    if val < minval: return minval
    if val > maxval: return maxval
    return val


def determine_gen_acc(num_of_sensors_transmitting):
    """
    With this function, we determine the number of objects we detected, using the mean and standard deviation we
    obtained from the Lidar dataset.

    Input: The number of sensors Transmitting
    Output: Recall value for the given frame
    """
    # detect based on parameters we pre-determined
    detected_objects = np.random.normal(GEN_MEAN[num_of_sensors_transmitting]*GEN_MAX_NUM_OBJECTS,
                                        GEN_SDV[num_of_sensors_transmitting] * GEN_MAX_NUM_OBJECTS, 1)
    # round the value and make sure it is within the borders
    detected_objects = restrict(np.round(detected_objects[0]), 0.0, GEN_MAX_NUM_OBJECTS)
    return np.round(detected_objects/GEN_MAX_NUM_OBJECTS, 3)


def decode_action_into_strategy_marl(action_value):
    """
    As the name suggest, we decode DQN's agent action into how the cameras should update in the next  time-step.

    Return list of actions: [mode_for_cam1, mode_for_cam2]
    """
    switcher = {
        0: MODE_STAND_BY,
        1: MODE_TRAN,
    }

    return switcher.get(action_value, "Selected action is NOT defined!!!")


def determine_latency(transmission_speed, data_to_transmit, num_frames):
    """
    A function that determines the latency based on the selected mode of operation. We follow the following Eqution:

    t_D = t_tr + t_dist + t_proc
    t_tr - transmission time: N_of_bits_to_transmit / Bandwidth
    t_dist - delays due to distance: distance / c_o (speed of light), distance< 500, overall time less than 10 < us
    t_proc - time process the image, this is constant as obtained in the paper we use to model the energy consumption,
             There are two different delays, one for processing on the edge device and the other on the cloudlet.

    """

    latency = data_to_transmit/transmission_speed + np.random.normal(LAT_CLOUDLET_PROC_TIME, LAT_CLOUDLET_PROC_TIME *
                                                                     LAT_VARY_CONSTANT) * num_frames

    return latency

