# coding=utf-8
import gym
import traci
import numpy as np
import xml.etree.ElementTree as ET

from abc import ABC
from sumolib import checkBinary
from parameters import *
from env.tls import Tls
from utils import load_json
from collections import OrderedDict
from maps.net_simu_generator import gen_cfg_file


class MATSC_Env(gym.Env, ABC):
    def __init__(self, server_number, test=False):
        super(MATSC_Env, self).__init__()

        self.test = test
        self.server_number = server_number

        self.config_data = None
        self.tls_list = None
        self.rl_tls_list = None
        self.non_rl_tls_list = None
        self.start_simulation_step = None

        self.env_params = SUMO_PARAMS

        self.gui = self.env_params.GUI
        self.seed = self.env_params.SEED
        self.co_train = self.env_params.CO_TRAIN
        self.random_seed = self.env_params.RANDOM_SEED
        self.obs_sharing = self.env_params.OBS_SHARING
        self.reward_detector = self.env_params.REWARD_DETECTOR
        self.regional_reward = self.env_params.REGIONAL_REWARD
        self.max_sumo_step = self.env_params.MAX_SUMO_STEP
        self.max_test_step = self.env_params.MAX_TEST_STEP
        self.sumo_max_distance = self.env_params.MAX_DISTANCE
        self.sumo_steps_green_phase = self.env_params.GREEN_DURATION
        self.sumo_steps_yellow_phase = self.env_params.YELLOW_DURATION
        self.sumo_teleport_time = self.env_params.TELEPORT_TIME

        self.all_datasets = self.env_params.ALL_DATASETS
        if self.co_train and self.server_number is not None and not self.test:
            self.net_file_name = self.all_datasets[self.server_number]
            self.net_file_path = './maps/{}'.format(self.net_file_name)
            self.net_config_path = self.net_file_path + '/{}_config.json'.format(self.net_file_name)
        else:
            self.net_file_name = self.env_params.NET_NAME
            self.net_file_path = self.env_params.NET_PATH
            self.net_config_path = self.env_params.CONFIG_PATH
        self.sumo_file_path = self.net_file_path + "/{}_{}.sumocfg".format(self.net_file_name, self.server_number)
        self.trip_file_path = self.net_file_path + "/trip_{}.xml".format(self.server_number)
        self.routes_output_file_path = self.net_file_path + "/routes_output_{}.xml".format(self.server_number)
        self.statistic_output_file_path = self.net_file_path + "/statistic_{}.xml".format(self.server_number)

        self.tls_dict = OrderedDict()
        self.incoming_lane_dict = OrderedDict()
        self.outgoing_lane_dict = OrderedDict()
        self.outgoing_lane_state_dict = OrderedDict()
        self.pre_actions_dict = OrderedDict()
        self.neighbor_dict = OrderedDict()
        self.tls_obs_space_n = OrderedDict()
        self.tls_action_space_n = OrderedDict()
        self.tls_action_group_n = OrderedDict()
        self.tls_obs_space = None
        self.tls_action_space = None
        self.tls_agent_space = None
        self.tls_all_agent_space = None
        self.tls_int_attr_space = None
        self.tls_max_movement_dim = None
        self.tls_max_phase_dim = None
        self.tls_movement_feat_dim = None
        self.tls_group_list = None
        self.tls_map_dataset = None
        self.tls_individual_input_dim = None
        self.tls_individual_action_dim = None
        self.tls_individual_group_input_dim = None
        self.tls_individual_group_action_dim = None
        self.tls_AttendLight_obs_space_n = None
        self.tls_AttendLight_action_space_n = None
        self.tls_AttendLight_feat_space_n = None
        self.tls_AttendLight_agent_space = None
        self.tls_AttendLight_max_num_parti_lane = None

        self.rl_step = 0
        self.sumo_step = 0
        self.curr_episode = 0
        self.trip_data = []
        self.traffic_data = []
        self.metrics_resco = []

        self.initialization()

    def initialization(self):
        """
        Initialize net configuration loader and load net configurations
        @return:
        """
        self.config_data = load_json(self.net_config_path)

        self.tls_list = self.rl_tls_list = list(self.config_data.keys())
        self.non_rl_tls_list = []

        all_incoming_lane_list, all_outgoing_lane_list = [], []
        for tls in self.tls_list:
            self.tls_dict[tls] = Tls(tls_id=tls, tls_config_data=self.config_data[tls])
            self.neighbor_dict[tls] = self.tls_dict[tls].neighbor_list
            self.incoming_lane_dict[tls] = self.tls_dict[tls].incoming_lane_list
            self.outgoing_lane_dict[tls] = self.tls_dict[tls].outgoing_lane_list
            all_incoming_lane_list += self.tls_dict[tls].incoming_lane_list
            all_outgoing_lane_list += self.tls_dict[tls].outgoing_lane_list

        for out_lane in all_outgoing_lane_list:
            if out_lane in all_incoming_lane_list:
                self.outgoing_lane_state_dict[out_lane] = 1
            else:
                self.outgoing_lane_state_dict[out_lane] = 0

        # For padding across all different scenarios, the maximum movement dimension is 36, the maximum phase dimension is 8
        if self.co_train:
            self.tls_max_movement_dim = 36
            self.tls_max_phase_dim = 8
            self.tls_all_agent_space = 97
        else:
            self.tls_max_movement_dim = np.max([len(self.tls_dict[tls].traffic_movement_list) for tls in self.tls_list])
            self.tls_max_phase_dim = np.max([len(self.tls_dict[tls].action_space) for tls in self.tls_list])
            self.tls_agent_space = len(self.tls_list)
            self.tls_all_agent_space = len(self.tls_list)
        # For Unicorn
        self.tls_agent_space = len(self.tls_list)
        self.tls_movement_feat_dim = 8
        self.tls_obs_space = [self.tls_max_movement_dim, self.tls_movement_feat_dim]
        self.tls_int_attr_space = 55

        # For IndividualLight
        self.tls_group_list = []
        self.tls_obs_space_n = OrderedDict()
        self.tls_action_space_n = OrderedDict()
        for tls in self.tls_list:
            self.tls_obs_space_n[tls] = (self.tls_max_movement_dim, self.tls_movement_feat_dim)
            self.tls_action_space_n[tls] = len(self.tls_dict[tls].action_space)
            if self.tls_dict[tls].phase_type not in self.tls_group_list:
                self.tls_group_list.append(self.tls_dict[tls].phase_type)

        self.tls_individual_input_dim = list(self.tls_obs_space_n.values())
        self.tls_individual_action_dim = list(self.tls_action_space_n.values())
        self.tls_individual_group_input_dim = []
        self.tls_individual_group_action_dim = []
        for phase_type in self.tls_group_list:
            self.tls_individual_group_input_dim.append((self.tls_max_movement_dim, self.tls_movement_feat_dim))
            self.tls_individual_group_action_dim.append(int(phase_type.split('.')[0]))

        # For AttendLight
        self.tls_AttendLight_feat_space_n = 3
        self.tls_AttendLight_max_num_parti_lane = 20
        self.tls_AttendLight_agent_space = len(self.tls_list)
        self.tls_AttendLight_obs_space_n = [self.tls_AttendLight_max_num_parti_lane, self.tls_AttendLight_feat_space_n]
        self.tls_AttendLight_action_space_n = self.tls_max_phase_dim

        if self.net_file_name == 'grid_network_5_5' or self.net_file_name == 'monaco_network_30':
            self.tls_map_dataset = 'ma2c'
        elif self.net_file_name == 'cologne_network_8' or self.net_file_name == 'ingolstadt_network_21' or self.net_file_name == 'grid_network_4_4' or self.net_file_name == 'arterial_network_4_4':
            self.tls_map_dataset = 'resco'
        elif self.net_file_name == 'shaoxing_network_7' or self.net_file_name == 'shenzhen_network_55' or self.net_file_name == 'shenzhen_network_29':
            self.tls_map_dataset = 'gesa'
        elif self.net_file_name == 'singapore_network_16':
            self.tls_map_dataset = 'sg'
        else:
            raise NotImplementedError

    def set_world(self, seed, write_unfinished=False):
        """
        Launch simulation
        @param seed: random seed for simulation
        @param write_unfinished: whether write the info of unfinished vehicles to the trip output
        @return:
        """
        # Generate new sumo configuration files when resetting the environment and launch new simulation
        gen_cfg_file(path=self.net_file_path, seed=seed, thread=self.server_number)

        try:
            if self.gui and (self.server_number == 0):
                sumoBinary = checkBinary('sumo-gui')
            else:
                sumoBinary = checkBinary('sumo')

            port = 6000 + self.server_number
            traci.start(
                [sumoBinary,
                 "--configuration-file", self.sumo_file_path,
                 "--tripinfo-output", self.trip_file_path,
                 "--tripinfo-output.write-unfinished", str(write_unfinished),
                 "--statistic-output", self.statistic_output_file_path,
                 "--duration-log.statistics", 'False',
                 "--start",
                 "--quit-on-end",
                 "--seed", str(seed),
                 "--no-warnings", 'True',
                 "--no-step-log", 'True',
                 "--time-to-teleport", str(self.sumo_teleport_time)],
                port=port)
            return 0

        except:

            print("Was not able to start simulation\n")
            return 1

    def reset(self):
        """
        Reset environment
        @return: Initial observation -> dictionary -> key/tls id, value/ np.array
        """
        print('Worker {} || Reset Environment'.format(self.server_number))
        # Set random seed for simulation
        if self.random_seed:
            self.seed = np.random.randint(0, 100000)
        else:
            self.seed += 1

        # Start SUMO simulation
        self.set_world(self.seed)
        # Force each agent to choose the first phase at the initial step
        for tls in self.tls_list:
            self.tls_dict[tls].set_green_phase(0, self.sumo_steps_green_phase)

        # Reset episode variables !
        self.reset_vars()

        return self.observe()

    def reset_test(self, curr_episode, seed):
        """
        Reset function for testing
        """
        assert self.test is True
        print('Worker {} || Reset Environment'.format(self.server_number))
        self.curr_episode = curr_episode

        # Start SUMO simulation
        self.set_world(seed=seed, write_unfinished=True)
        # Force each agent to choose the first phase at the initial step
        for tls in self.tls_list:
            self.tls_dict[tls].set_green_phase(0, self.sumo_steps_green_phase)

        # Reset episode variables !
        self.reset_vars()

        return self.observe()

    def reset_AttendLight(self):
        """
        Reset environment
        @return: Initial observation -> dictionary -> key/tls id, value/ np.array
        """
        print('Worker {} || Reset Environment'.format(self.server_number))
        # Set random seed for simulation
        if self.random_seed:
            self.seed = np.random.randint(0, 100000)
        else:
            self.seed += 1

        # Start SUMO simulation
        self.set_world(self.seed)
        # Force each agent to choose the first phase at the initial step
        for tls in self.tls_list:
            self.tls_dict[tls].set_green_phase(0, self.sumo_steps_green_phase)

        # Reset episode variables !
        self.reset_vars()

        return self.observe_AttendLight()

    def reset_vars(self):
        """
        Reset variables
        """
        for tls in self.tls_list:
            self.pre_actions_dict[tls] = None
        self.rl_step = 0
        self.sumo_step = 0
        self.start_simulation_step = traci.simulation.getTime()

    def step(self, action_dict):
        """
        Interact with environment by executing actions
        @param action_dict: dictionary, key/tls id, value/action/phase index
        @return: next observation -> dictionary -> key/tls id, value/np.array
        @return: reward -> dictionary -> key/tls id, value/float
        @return: done -> bool -> terminal
        @return: data: 1. Average reward for the whole net
        @return: data: 2. Average number of action change
        @return: data: 3. Traffic metric measurements
        """
        change_action_list = []
        sum_action_change = 0

        for tls in self.tls_list:
            if self.pre_actions_dict[tls] is not None and self.pre_actions_dict[tls] != action_dict[tls]:
                change_action_list.append(tls)
                if tls in self.rl_tls_list:
                    sum_action_change += 1

        # Step a multi-agent rl step
        for t in range(self.sumo_steps_green_phase):
            for tls in self.tls_list:
                if t == 0:
                    if tls in change_action_list:
                        self.tls_dict[tls].set_yellow_phase(pre_action=self.pre_actions_dict[tls],
                                                            duration=self.sumo_steps_yellow_phase)
                    else:
                        self.tls_dict[tls].set_green_phase(action_dict[tls],
                                                           duration=self.sumo_steps_green_phase)

                elif t == self.sumo_steps_yellow_phase:
                    if tls in change_action_list:
                        self.tls_dict[tls].set_green_phase(action=action_dict[tls],
                                                           duration=(self.sumo_steps_green_phase -
                                                                     self.sumo_steps_yellow_phase))
            self.step_a_sumo_step()

        self.rl_step += 1
        self.pre_actions_dict = action_dict

        next_obs_dict = self.observe()
        rewards_dict = self.calculate_reward()
        done = self.check_terminal()
        avg_action_change = sum_action_change / int(len(self.rl_tls_list))
        avg_multi_agent_reward = sum(rewards_dict.values()) / int(len(self.rl_tls_list))

        return next_obs_dict, rewards_dict, done, [avg_multi_agent_reward, avg_action_change]

    def step_AttendLight(self, action_dict):
        """
        Interact with environment by executing actions
        @param action_dict: dictionary, key/tls id, value/action/phase index
        @return: next observation -> dictionary -> key/tls id, value/np.array
        @return: reward -> dictionary -> key/tls id, value/float
        @return: done -> bool -> terminal
        @return: data: 1. Average reward for the whole net
        @return: data: 2. Average number of action change
        @return: data: 3. Traffic metric measurements
        """
        change_action_list = []
        sum_action_change = 0

        for tls in self.tls_list:
            if self.pre_actions_dict[tls] is not None and self.pre_actions_dict[tls] != action_dict[tls]:
                change_action_list.append(tls)
                if tls in self.rl_tls_list:
                    sum_action_change += 1

        # Step a multi-agent rl step
        for t in range(self.sumo_steps_green_phase):
            for tls in self.tls_list:
                if t == 0:
                    if tls in change_action_list:
                        self.tls_dict[tls].set_yellow_phase(pre_action=self.pre_actions_dict[tls],
                                                            duration=self.sumo_steps_yellow_phase)
                    else:
                        self.tls_dict[tls].set_green_phase(action_dict[tls], duration=self.sumo_steps_green_phase)

                elif t == self.sumo_steps_yellow_phase:
                    if tls in change_action_list:
                        self.tls_dict[tls].set_green_phase(action=action_dict[tls],
                                                           duration=(self.sumo_steps_green_phase -
                                                                     self.sumo_steps_yellow_phase))
            self.step_a_sumo_step()

        self.rl_step += 1
        self.pre_actions_dict = action_dict

        next_obs = self.observe_AttendLight()
        rewards_dict = self.calculate_reward()
        done = self.check_terminal()
        avg_action_change = sum_action_change / int(len(self.rl_tls_list))
        avg_multi_agent_reward = sum(rewards_dict.values()) / int(len(self.rl_tls_list))

        return next_obs, rewards_dict, done, [avg_multi_agent_reward, avg_action_change]

    def step_a_sumo_step(self):
        """
        Interacting with sumo
        @return:
        """
        self.sumo_step += 1
        traci.simulationStep()
        if self.test:
            self.measure_traffic_step()

    def observe(self):
        """
        Interact with environment and get observations
        @return: dictionary, key/tls id, value/np.array
        """
        single_observations = OrderedDict()
        for tls in self.rl_tls_list:
            # obs = self.tls_dict[tls].observe(max_distance=self.sumo_max_distance)
            if self.tls_map_dataset == 'ma2c':
                obs = self.tls_dict[tls].observe_ma2c_network()
            elif self.tls_map_dataset == 'resco':
                obs = self.tls_dict[tls].observe_resco_network(out_lane_state_dict=self.outgoing_lane_state_dict)
            elif self.tls_map_dataset == 'gesa':
                obs = self.tls_dict[tls].observe_ma2c_network()
            elif self.tls_map_dataset == 'sg':
                obs = self.tls_dict[tls].observe_ma2c_network()
            else:
                raise NotImplementedError
            pad_dim = self.tls_max_movement_dim - obs.shape[0]
            # padding the single observation to the dimension of [max_num_movement, movement_feat]
            single_observations[tls] = np.concatenate((obs, np.zeros((pad_dim, obs.shape[1]))), 0)

        if self.obs_sharing:
            total_observations = {}
            for tls in self.rl_tls_list:
                combined_observation = [single_observations[tls]]
                neighbor_list = self.tls_dict[tls].neighbor_list
                for neighbor_id in neighbor_list:
                    if neighbor_id is None:
                        combined_observation.append(np.zeros(single_observations[tls].copy().shape))
                    else:
                        combined_observation.append(single_observations[neighbor_id])

                total_observations[tls] = np.array(combined_observation)
            return total_observations
        else:
            return single_observations

    def observe_AttendLight(self):
        """
        Get the observation vectors for each intersection and apply padding for batch training:
        The observation dimension is [max_num_phase, max_num_parti_lane, lane_feat_dim]
        return: dictionary, key/tls id, value/np.arrays (observation vectors, state mask vectors, action mask vectors)
        """
        single_observations = OrderedDict()
        state_mask_dict = OrderedDict()
        action_mask_dict = OrderedDict()
        curr_phase_index_dict = self.get_current_phase_index_AttendLight()
        for tls in self.rl_tls_list:
            # get raw observation without padding
            if self.tls_map_dataset == 'ma2c' or self.tls_map_dataset == 'gesa':
                observation = self.tls_dict[tls].observe_AttendLight_ma2c_network()
            elif self.tls_map_dataset == 'resco':
                observation = self.tls_dict[tls].observe_AttendLight_resco_network()
            elif self.tls_map_dataset == 'sg':
                observation = self.tls_dict[tls].observe_AttendLight_resco_network()
            else:
                observation = self.tls_dict[tls].observe_AttendLight()
            state_mask = []
            for i, vec in enumerate(observation):
                lane_pad_dim = self.tls_AttendLight_max_num_parti_lane - len(vec)
                mask = [0] * len(vec) + [1] * lane_pad_dim
                state_mask.append(mask)
                # padding the features of participant lanes for each phase
                observation[i] += [np.zeros_like((1, 1, self.tls_AttendLight_feat_space_n)).tolist()] * lane_pad_dim

            phase_pad_dim = self.tls_max_phase_dim - len(observation)
            action_mask = [0] * len(observation) + [1] * phase_pad_dim
            # padding the single observation to the dimension of [max_num_phase, max_num_parti_lanes]
            for _ in range(phase_pad_dim):
                observation.append(np.zeros_like(observation[0]).tolist())
                state_mask.append(np.zeros(self.tls_AttendLight_max_num_parti_lane, dtype=int).tolist())

            single_observations[tls] = observation
            state_mask_dict[tls] = state_mask
            action_mask_dict[tls] = action_mask

        return [single_observations, state_mask_dict, curr_phase_index_dict, action_mask_dict]

    def get_current_phase_index_AttendLight(self):
        """
        Get the current activated phase index for each intersection
        """
        activated_phase_index_dict = OrderedDict()
        for tls in self.rl_tls_list:
            activated_phase = traci.trafficlight.getRedYellowGreenState(tls)
            if activated_phase not in self.tls_dict[tls].action_space:
                traci.trafficlight.setRedYellowGreenState(tls, self.tls_dict[tls].action_space[0])
                activated_phase_index_dict[tls] = 0
            else:
                activated_phase_index_dict[tls] = self.tls_dict[tls].action_space.index(activated_phase)
        return activated_phase_index_dict

    def get_phase_vec_mask_dict(self):
        """
        Get the phase vectors and phase masks for parameter-sharing learning
        """
        phase_vec_dict = OrderedDict()
        phase_mask_dict = OrderedDict()
        for tls in self.rl_tls_list:
            phase_vec = np.array(list(self.tls_dict[tls].phase_movement_dict.values()))
            phase_mask_dict[tls] = np.ones(self.tls_max_phase_dim)
            phase_mask_dict[tls][:phase_vec.shape[0]] = 0

            # padding the phase vec to the dimension of [max_num_phase, max_num_movement]
            pad_dim_1 = self.tls_max_movement_dim - phase_vec.shape[1]
            pad_dim_2 = self.tls_max_phase_dim - phase_vec.shape[0]
            # padding for axis 1: traffic movement dimension
            phase_vec = np.concatenate((phase_vec, np.zeros((phase_vec.shape[0], pad_dim_1))), -1)
            # padding for axis 0: phase space dimension
            phase_vec = np.concatenate((phase_vec, np.zeros((pad_dim_2, phase_vec.shape[1]))), 0)
            phase_vec_dict[tls] = phase_vec

        return phase_vec_dict, phase_mask_dict

    def get_int_attr_vec_dict(self, padding=True):
        """
        Get the intersection attribute vectors for each intersection
        """
        int_attr_vec_dict = OrderedDict()
        for tls in self.rl_tls_list:
            if padding:
                int_attr_vec_dict[tls] = np.array([self.tls_dict[tls].int_attr_vec] * self.tls_max_phase_dim)
            else:
                int_attr_vec_dict[tls] = self.tls_dict[tls].int_attr_vec

        return int_attr_vec_dict

    def get_neighbor_actions(self, action_dict):
        """
        Retrieve the actions/chosen phases of neighbors. In heterogeneous networks, the situation becomes complex due to the
        presence of non-signalized intersections between learning agents. To address this, we model the neighbor's currently chosen action
        based on the activation status of ego outgoing lanes. This approach is independent of the phase settings of neighbors
        with varying topologies. For non-signalized intersections, outgoing lanes are consistently marked as active.
        """
        # Calculate the activation status of the controlled lanes
        lane_activation_dict = dict()
        for tls, action in action_dict.items():
            phase_code = self.tls_dict[tls].valid_phase_list[action]
            for i, code in enumerate(phase_code):
                in_lane, _ = self.tls_dict[tls].traffic_movement_list[i]
                # Check the activation status of the given lane
                if code == 'G':
                    status = 1
                elif code == 'g':
                    status = 0.5
                else:
                    status = 0

                if in_lane not in lane_activation_dict:
                    lane_activation_dict[in_lane] = [status]
                else:
                    lane_activation_dict[in_lane].append(status)

        # Since a lane might participate in multiple movements, thus a lane can have different activation status
        for lane, status_list in lane_activation_dict.items():
            lane_activation_dict[lane] = np.mean(status_list)

        # Calculate neighbors' actions
        neighbors_action_dict = OrderedDict()
        for tls in self.tls_list:
            neighbor_action = []
            for i, movement in enumerate(self.tls_dict[tls].traffic_movement_list):
                _, out_lane = movement
                if out_lane in lane_activation_dict:
                    neighbor_action.append(lane_activation_dict[out_lane])
                else:
                    neighbor_action.append(1)

            padding_dim = self.tls_max_movement_dim - len(self.tls_dict[tls].traffic_movement_list)
            for _ in range(padding_dim):
                neighbor_action.append(0)

            neighbors_action_dict[tls] = neighbor_action

        return neighbors_action_dict

    def calculate_reward(self):
        """
        Calculate rewards
        @return: dictionary, key/Tls id, value/reward
        Note: if share reward with neighbors, return average reward combined with neighbours
        """
        single_rewards = OrderedDict()
        for tls in self.rl_tls_list:
            # single_rewards[tls] = self.tls_dict[tls].get_truncated_queue_reward(detector=self.reward_detector)
            if self.tls_map_dataset == 'ma2c':
                single_rewards[tls] = self.tls_dict[tls].get_truncated_queue_reward_ma2c_network(regional_reward=self.regional_reward)
            elif self.tls_map_dataset == 'resco':
                single_rewards[tls] = self.tls_dict[tls].get_truncated_queue_reward_resco_network(regional_reward=self.regional_reward)
            elif self.tls_map_dataset == 'gesa':
                single_rewards[tls] = self.tls_dict[tls].get_truncated_queue_reward_ma2c_network(regional_reward=self.regional_reward)
            elif self.tls_map_dataset == 'sg':
                single_rewards[tls] = self.tls_dict[tls].get_truncated_queue_reward_ma2c_network(regional_reward=self.regional_reward)
            else:
                raise NotImplementedError

        return single_rewards

    def calculate_regional_reward(self):
        """
        Calculate regional rewards for coordination
        @return: dictionary, key/Tls id, value/reward
        """
        single_rewards = OrderedDict()
        for tls in self.rl_tls_list:
            single_rewards[tls] = self.tls_dict[tls].get_regional_queue_reward(distance=50, out_queue_weight=1)

        return single_rewards

    def check_terminal(self):
        """
        Terminal condition:
        1. Test: reach the maximum sumo step
        2. Non-test: reach the maximum sumo step or all vehicles reach destinations
        @return: done
        """
        done = False
        if self.test:
            if (traci.simulation.getTime() - self.start_simulation_step) >= self.max_test_step:
                done = True
        else:
            if (traci.simulation.getTime() - self.start_simulation_step) >= self.max_sumo_step:
                done = True
            # elif traci.simulation.getMinExpectedNumber() <= 0:
            #     done = True

        if done:
            traci.close()

        return done

    def measure_traffic_step(self):
        """
        Measure traffic metrics per sumo step only for evaluation/test
        @return: dictionary, key/metric name, value/metric value
        1. avg_speed_mps: average speed for the whole network
        2. avg_wait_sec: average waiting time for the whole network
        3. avg_queue: average queue for the whole network
        4. std_queue: stand deviation of queue for the whole network
        @warning: activating this function will slow down the simulation
        """
        veh_list = traci.vehicle.getIDList()
        num_tot_car = len(veh_list)
        num_in_car = traci.simulation.getDepartedNumber()
        num_out_car = traci.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(car) for car in veh_list])
            avg_speed = np.mean([traci.vehicle.getSpeed(car) for car in veh_list])
        else:
            avg_speed = 0
            avg_waiting_time = 0

        # all trip-related measurements are not supported by traci
        queues = []
        for tls in self.tls_dict.keys():
            for in_lane in self.tls_dict[tls].incoming_lane_list:
                queues.append(traci.lane.getLastStepHaltingNumber(in_lane))
        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))

        curr_traffic = {'episode': self.curr_episode,
                        'time_sec': traci.simulation.getTime(),
                        'number_total_car': num_tot_car,
                        'number_departed_car': num_in_car,
                        'number_arrived_car': num_out_car,
                        'avg_wait_sec': avg_waiting_time,
                        'avg_speed_mps': avg_speed,
                        'std_queue': std_queue,
                        'avg_queue': avg_queue}

        self.traffic_data.append(curr_traffic)

        return curr_traffic

    def calculate_target_queue(self, lane=True):
        """
        calculate all target queue values for predictions
        @return: dictionary, key/tls_id, value/target queue values
        """
        truth_queue = {}
        for tls_id in self.rl_tls_list:
            if lane:
                truth_queue[tls_id] = self.tls_dict[tls_id].calculate_target_queue(norm=True, lane=True)
            else:
                truth_queue[tls_id] = self.tls_dict[tls_id].calculate_target_queue(norm=True, lane=False)

        return truth_queue

    def calculate_queue_vectors(self):
        """
        Calculate queue vectors together with neighbor intersections
        @return:
        """
        truth_queue = {}
        for tls_id in self.rl_tls_list:
            truth_queue[tls_id] = self.tls_dict[tls_id].calculate_target_queue(norm=True, lane=True)

        queue_vectors_neighbor = {}
        for tls_id in self.rl_tls_list:
            queue_vectors_neighbor[tls_id] = [truth_queue[tls_id]]
            for neighbor in self.tls_dict[tls_id].neighbor_list:
                if neighbor is not None:
                    queue_vectors_neighbor[tls_id].append(truth_queue[neighbor])
                else:
                    queue_vectors_neighbor[tls_id].append(np.zeros_like(truth_queue[tls_id]).tolist())
            # Convert to numpy array
            queue_vectors_neighbor[tls_id] = np.array(queue_vectors_neighbor[tls_id])

        return queue_vectors_neighbor

    def measure_perf_episode(self):
        """
        Read the data from statistic output to get the episode performance metrics
        @return dict:
        1. Average speed
        2. Average duration (trip time)
        3. Average waiting time
        4. Average time loss (due to deceleration/acceleration)
        5. Average travel time (plus non-inserted vehicles)
        """
        data = {}
        tree = ET.ElementTree(file=self.statistic_output_file_path)
        for child in tree.getroot():
            data[child.tag] = child.attrib

        # The real travel should be calculated by two parts:
        # 1. The travel time of inserted vehicles
        # 2. The travel time of non-inserted vehicles
        travel_time = float(data['vehicles']['inserted']) * \
                      (float(data['vehicleTripStatistics']['duration']) + float(
                          data['vehicleTripStatistics']['departDelay'])) \
                      + float(data['vehicles']['waiting']) * float(data['vehicleTripStatistics']['departDelayWaiting'])

        perf = {
            'avg_speed': float(data['vehicleTripStatistics']['speed']),
            'avg_duration': float(data['vehicleTripStatistics']['duration']),
            'avg_wait_time': float(data['vehicleTripStatistics']['waitingTime']),
            'avg_time_loss': float(data['vehicleTripStatistics']['timeLoss']),
            'avg_travel_time': travel_time / float(data['vehicles']['loaded'])
        }
        return perf

    def collect_trip_data(self):
        """
        Collect all trip information for each episode after simulation
        @return:
        """
        tree = ET.ElementTree(file=self.trip_file_path)
        for child in tree.getroot():
            curr_trip = child.attrib
            curr_dict = {'episode': int(self.curr_episode),
                         'id': curr_trip['id'],
                         'depart_sec': str(float(curr_trip['depart']) - self.start_simulation_step),
                         'arrival_sec': str(float(curr_trip['arrival']) - self.start_simulation_step),
                         'duration_sec': curr_trip['duration'],
                         'wait_sec': curr_trip['waitingTime'],
                         'wait_step': curr_trip['waitingCount'],
                         'time_loss_sec': curr_trip['timeLoss']}
            # Add the trip data to the list
            self.trip_data.append(curr_dict)

    def calculate_metrics_resco(self, max_distance=200):
        """
        Calculate traffic metrics (queue length at each intersection) at each decision time step for evaluation.
        In RESCO codes, the undetectable vehicles will be removed from the lane (by default the detectable distance is 200)
        """
        def get_vehicles(lane, max_distance):
            detectable = []
            for vehicle in traci.lane.getLastStepVehicleIDs(lane):
                path = traci.vehicle.getNextTLS(vehicle)
                if len(path) > 0:
                    next_light = path[0]
                    distance = next_light[2]
                    if distance <= max_distance:  # Detectors have a max range
                        detectable.append(vehicle)
            return detectable

        queue_lengths = dict()
        max_queues = dict()
        for tls in self.tls_list:
            queue_length, max_queue = 0, 0
            for in_lane in self.tls_dict[tls].incoming_lane_list:
                # queue = signal.full_observation[lane]['queue']
                queue = 0
                lane_vehicles = get_vehicles(in_lane, max_distance)
                for vehicle in lane_vehicles:
                    if traci.vehicle.getWaitingTime(vehicle) > 0:
                        queue += 1
                if queue > max_queue:
                    max_queue = queue
                queue_length += queue
            queue_lengths[tls] = queue_length
            max_queues[tls] = max_queue
        self.metrics_resco.append({
            'step': traci.simulation.getTime(),
            'reward': rewards,
            'max_queues': max_queues,
            'queue_lengths': queue_lengths
        })

    def save_metrics_resco(self, save_file_path):
        with open(save_file_path, 'w+') as output_file:
            for line in self.metrics_resco:
                csv_line = ''
                for metric in ['step', 'reward', 'max_queues', 'queue_lengths']:
                    csv_line = csv_line + str(line[metric]) + ', '
                output_file.write(csv_line + '\n')


if __name__ == '__main__':
    import os

    os.chdir('../')
    env = MATSC_Env(server_number=0)
    obs_n = env.reset()
    done = False
    # max_length = 0
    # for tls in env.tls_list:
    #     for in_lane in env.tls_dict[tls].incoming_lane_list:
    #         length = traci.lane.getLength(in_lane)
    #         if length > max_length:
    #             max_length = length
    #     for out_lane in env.tls_dict[tls].outgoing_lane_list:
    #         length = traci.lane.getLength(out_lane)
    #         if length > max_length:
    #             max_length = length
    # print(max_length)
    while not done:
        action_dict = {}
        for id in env.tls_list:
            # action_dict[id] = np.random.randint(0, len(env.tls_dict[id].action_space))
            # action_dict[id] = env.tls_dict[id].get_fixed_time_action()
            # action_dict[id] = env.tls_dict[id].get_greedy_action()
            action_dict[id] = env.tls_dict[id].get_pressure_action()
        print('Time:{}'.format(traci.simulation.getTime()))
        next_obs, rewards, done, info = env.step(action_dict=action_dict)
