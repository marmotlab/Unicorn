import traci
import numpy as np

from collections import OrderedDict


class Tls:
    def __init__(self, tls_id, tls_config_data):
        """

        @rtype: object
        """
        self.tls_id = tls_id
        self.config_data = tls_config_data

        self.incoming_lane_list = None
        self.outgoing_lane_list = None
        self.incoming_detector_list = None
        self.outgoing_detector_list = None
        self.lane_links = None
        self.num_links_lane = None
        self.action_space = None
        self.neighbor_list = None
        self.phase_lane_state = None
        self.phase_type = None
        self.phase_type_vec = None
        self.valid_phase_list = None
        self.tls_position_vec = None
        self.action_space_n = None
        self.num_in_lane = None
        self.num_out_lane = None
        self.node_list = None
        self.traffic_movement_list = None
        self.phase_connection_matrix = None
        self.node_connection_dict = None
        self.phase_movement_dict = None
        self.incoming_edge_list = None
        self.outgoing_edge_list = None
        self.int_attr_vec = None
        self.lane_length_dict = None
        self.lane_max_speed_dict = None
        self.replaced_lane_dict = None
        self.replaced_lane_length_dict = None
        self.lane_detector_dict = None

        self.lane_state_one_hot = None
        self.num_veh_in_lane = None
        self.num_veh_out_lane = None
        self.lane_area_occupancy = None
        self.waiting_time_front_veh = None
        self.num_veh_in_detector = None
        self.pre_veh_list_in_lane = None
        self.pre_veh_list_out_lane = None

        self.num_veh_seg_1 = None
        self.num_veh_seg_2 = None
        self.num_veh_seg_3 = None
        self.avg_speed_seg_1 = None
        self.avg_speed_seg_2 = None
        self.avg_speed_seg_3 = None

        # Hardcoded
        self.num_segment = 3
        self.max_num_veh_seg = 8
        self.seg_pos = [60, 120, 180]
        self.initialization()

    def initialization(self):
        """
        To load the agent attributes/intersection configurations
        Must call it when create instances
        """
        self.incoming_lane_list = self.config_data['incoming_lane_list']
        self.outgoing_lane_list = self.config_data['outgoing_lane_list']
        self.incoming_detector_list = self.config_data['incoming_detector_list']
        self.outgoing_detector_list = self.config_data['outgoing_detector_list']
        self.lane_links = self.config_data['lane_links']
        self.num_links_lane = self.config_data['num_links_lane']
        self.action_space = self.config_data['action_space']
        self.neighbor_list = self.config_data['neighbor_list']
        self.phase_lane_state = self.config_data['phase_lane_state']
        self.phase_type = self.config_data['phase_type']
        self.phase_type_vec = self.config_data['phase_type_vec']
        self.tls_position_vec = self.config_data['tls_position_vec']
        self.lane_length_dict = self.config_data['lane_length_dict']
        self.lane_max_speed_dict = self.config_data['lane_max_speed_dict']
        self.valid_phase_list = self.config_data['valid_phase_list']

        # For Ingolstadt maps mainly
        if 'replaced_lane_dict' in self.config_data:
            self.replaced_lane_dict = self.config_data['replaced_lane_dict']
            self.replaced_lane_length_dict = self.config_data['replaced_lane_length_dict']

        else:
            self.replaced_lane_dict = dict()
            self.replaced_lane_length_dict = dict()

        # intersection attributes
        self.action_space_n = len(self.action_space)
        self.num_in_lane = len(self.incoming_lane_list)
        self.num_out_lane = len(self.outgoing_lane_list)

        self.incoming_edge_list = []
        self.outgoing_edge_list = []
        self.pre_veh_list_in_lane = dict()
        self.lane_detector_dict = dict()
        for i, in_lane in enumerate(self.incoming_lane_list):
            self.pre_veh_list_in_lane[i] = []
            # in_edge = in_lane.split('_')[0]
            in_edge = in_lane[:-2]
            if in_edge not in self.incoming_edge_list:
                self.incoming_edge_list.append(in_edge)
            self.lane_detector_dict[in_lane] = self.incoming_detector_list[i]

        self.pre_veh_list_out_lane = dict()
        for i, out_lane in enumerate(self.outgoing_lane_list):
            self.pre_veh_list_out_lane[i] = []

        for i, out_lane in enumerate(self.outgoing_lane_list):
            # out_edge = out_lane.split('_')[0]
            out_edge = out_lane[:-2]
            if out_edge not in self.outgoing_edge_list:
                self.outgoing_edge_list.append(out_edge)
            self.lane_detector_dict[out_lane] = self.outgoing_detector_list[i]

        # self.get_node_connection()
        self.get_traffic_movements_phase()
        self.get_intersection_attributes()

    def get_node_connection(self):
        # Calculate the connection relationship between neighboring agents
        # data structure: key/node, value/dict: key/connected node, value/list of movement (in lane and out lane pair)
        self.node_list = []
        self.traffic_movement_list = []
        self.node_connection_dict = {}
        for in_lane, out_lanes in self.lane_links.items():
            in_node = in_lane.split('_')[0]
            if in_node not in self.node_connection_dict:
                self.node_connection_dict[in_node] = OrderedDict()
                self.node_list.append(in_node)
            for out_lane in out_lanes:
                self.traffic_movement_list.append((in_lane, out_lane))
                out_node = out_lane.split('_')[1]
                if out_node not in self.node_connection_dict[in_node]:
                    self.node_connection_dict[in_node][out_node] = [(in_lane, out_lane)]
                else:
                    self.node_connection_dict[in_node][out_node].append((in_lane, out_lane))

        # get the node connection matrix for each phase
        """
        row: in node, column: out node
        for instance: phase1 for N-S left-turn, right-turn and straight
              node1 node2 node3 node4
        node1   0     0     1     0
        node2  0.5    0     1     0
        node3   1     0     0     0
        node4   1     0    0.5    0
        """
        self.phase_connection_matrix = OrderedDict()
        for phase_code in self.action_space:
            self.phase_connection_matrix[phase_code] = np.zeros((len(self.node_list), len(self.node_list)))
            for i, code in enumerate(phase_code):
                if code == 'G' or code == 'g':
                    movement = self.traffic_movement_list[i]
                    in_lane, out_lane = movement
                    in_node = in_lane.split('_')[0]
                    out_node = out_lane.split('_')[1]
                    if code == 'G':
                        self.phase_connection_matrix[phase_code][self.node_list.index(in_node)][
                            self.node_list.index(out_node)] = 1
                    else:
                        self.phase_connection_matrix[phase_code][self.node_list.index(in_node)][
                            self.node_list.index(out_node)] = 0.5

                else:
                    pass
        return 0

    def get_traffic_movements_phase(self):
        self.traffic_movement_list = []
        for in_lane, out_lanes in self.lane_links.items():
            for out_lane in out_lanes:
                self.traffic_movement_list.append((in_lane, out_lane))

        self.phase_movement_dict = OrderedDict()
        for phase in self.valid_phase_list:
            phase_vec = []
            for code in phase:
                if code == 'G':
                    phase_vec.append(1)
                elif code == 'g':
                    phase_vec.append(0.5)
                else:
                    phase_vec.append(0)

            self.phase_movement_dict[phase] = phase_vec

    def get_intersection_attributes(self, max_num_edge=7):
        """
        Retrieve the constant attributes of the given intersection, which contains the information of:
        1. The type of the traffic light and traffic light phases (a binary vector)
        2. The average lane length of at incoming edges of the intersection
        3. The average maximum speed at incoming edges of the intersection
        4. The average number of lanes at incoming edges of the intersection
        5. The average number of links at the incoming edges of the intersection
        6. The average lane length at the outgoing edges of the intersection
        7. The average maximum speed at the outgoing edges of the intersection
        8. The average number lanes at the outgoing edges of the intersection
        """
        # Padding hexadecimal codes to 6 dimension vectors
        phase_type_vec = list([0] * (6 - len(self.phase_type_vec))) + self.phase_type_vec
        # tls_pos = self.tls_position_vec
        # set the maximum number of edges to 7
        in_edge_mean_length = []
        in_edge_mean_speed = []
        in_edge_num_lane = []
        in_edge_num_links = []
        for in_edge in self.incoming_edge_list:
            length, speed, num_lane, num_links = [], [], 0, 0
            for in_lane in self.incoming_lane_list:
                if in_edge in in_lane:
                    if in_lane in self.replaced_lane_dict:
                        length.append(self.replaced_lane_length_dict[in_lane])
                    else:
                        length.append(self.lane_length_dict[in_lane])

                    speed.append(self.lane_max_speed_dict[in_lane])
                    num_lane += 1
                    num_links += len(self.lane_links[in_lane])

            in_edge_mean_length.append(np.mean(length))
            in_edge_mean_speed.append(np.mean(speed))
            in_edge_num_lane.append(num_lane)
            in_edge_num_links.append(num_links)

        in_padding_dim = max_num_edge - len(self.incoming_edge_list)
        in_edge_mean_length += [0] * in_padding_dim
        in_edge_mean_speed += [0] * in_padding_dim
        in_edge_num_lane += [0] * in_padding_dim
        in_edge_num_links += [0] * in_padding_dim

        out_edge_mean_length = []
        out_edge_mean_speed = []
        out_edge_num_lane = []
        for out_edge in self.outgoing_edge_list:
            length, speed, num_lane, num_links = [], [], 0, 0
            for out_lane in self.outgoing_lane_list:
                if out_edge in out_lane:
                    if out_lane in self.replaced_lane_dict:
                        length.append(self.replaced_lane_length_dict[out_lane])
                    else:
                        length.append(self.lane_length_dict[out_lane])
                    speed.append(self.lane_max_speed_dict[out_lane])
                    num_lane += 1

            out_edge_mean_length.append(np.mean(length))
            out_edge_mean_speed.append(np.mean(speed))
            out_edge_num_lane.append(num_lane)

        out_padding_dim = max_num_edge - len(self.outgoing_edge_list)
        out_edge_mean_length += [0] * out_padding_dim
        out_edge_mean_speed += [0] * out_padding_dim
        out_edge_num_lane += [0] * out_padding_dim

        self.int_attr_vec = np.concatenate([np.array(phase_type_vec),
                                            np.array(in_edge_mean_length) / 300,
                                            np.array(in_edge_mean_speed) / 30,
                                            np.array(in_edge_num_lane) / 8,
                                            np.array(in_edge_num_links) / 8,
                                            np.array(out_edge_mean_length) / 300,
                                            np.array(out_edge_mean_speed) / 30,
                                            np.array(out_edge_num_lane) / 8])

    def set_yellow_phase(self, pre_action, duration):
        pre_green_code = self.action_space[pre_action]
        phase_code = []
        for i in pre_green_code:
            if i == 'r':
                phase_code.append(i)
            else:
                phase_code.append('y')

        phase_code = ''.join(phase_code)
        traci.trafficlight.setRedYellowGreenState(self.tls_id, phase_code)
        traci.trafficlight.setPhaseDuration(self.tls_id, duration)

    def set_green_phase(self, action, duration):
        phase_code = self.action_space[action]
        traci.trafficlight.setRedYellowGreenState(self.tls_id, phase_code)
        traci.trafficlight.setPhaseDuration(self.tls_id, duration)

    def observe_nodes(self):
        """
        Retrieve observation vectors based on phase node connection matrix. For each connection, the observation is defined as:
        1. The number of incoming lanes
        2. The number of outgoing lanes
        3. The mean and variance of length of incoming lanes connecting in nodes and out nodes
        4. The mean and variance of length of outgoing lanes connection in nodes and out nodes
        5. The mean and variance of the number of vehicles on the incoming lanes
        6. The mean and variance of the number of vehicles on the outgoing lanes
        7. The mean and variance of the number of halting vehicles on the incoming lanes
        8. The mean and variance of the number of halting vehicles on the outgoing lanes
        """
        observation = []
        for i, in_node in enumerate(self.node_list):
            observation.append([])
            for j, out_node in enumerate(self.node_list):
                if in_node != out_node:
                    movement_list = self.node_connection_dict[in_node][out_node]
                    in_lanes_set, out_lanes_set = set(), set()
                    for movement in movement_list:
                        in_lanes_set.add(movement[0])
                        out_lanes_set.add(movement[1])
                    num_in_lane, num_out_lane = len(in_lanes_set), len(out_lanes_set)
                    (length_in_lane_list, length_out_lane_list, num_veh_in_lane_list,
                     num_veh_out_lane_list, num_halting_veh_in_lane_list,
                     num_halting_veh_out_lane_list) = [], [], [], [], [], []

                    for in_lane in list(in_lanes_set):
                        length_in_lane_list.append(traci.lane.getLength(in_lane))
                        num_veh_in_lane_list.append(traci.lane.getLastStepVehicleNumber(in_lane))
                        num_halting_veh_in_lane_list.append(np.clip(traci.lane.getLastStepHaltingNumber(in_lane), 0, 8))

                    for out_lane in list(out_lanes_set):
                        length_out_lane_list.append(traci.lane.getLength(out_lane))
                        num_veh_out_lane_list.append(traci.lane.getLastStepVehicleNumber(out_lane))
                        num_halting_veh_out_lane_list.append(
                            np.clip(traci.lane.getLastStepHaltingNumber(out_lane), 0, 8))

                    length_in_mean = np.mean(length_in_lane_list)
                    num_veh_in_mean = np.mean(num_veh_in_lane_list)
                    num_halting_veh_in_mean = np.mean(num_halting_veh_in_lane_list)

                    length_out_mean = np.mean(length_out_lane_list)
                    num_veh_out_mean = np.mean(num_veh_out_lane_list)
                    num_halting_veh_out_lane_mean = np.mean(num_halting_veh_out_lane_list)

                    connection_feature = [num_in_lane / 10,
                                          num_out_lane / 10,
                                          length_in_mean / 200,
                                          length_out_mean / 200,
                                          num_veh_in_mean / 24,
                                          num_veh_out_mean / 24,
                                          num_halting_veh_in_mean / 10,
                                          num_halting_veh_out_lane_mean / 10]
                else:
                    connection_feature = [0] * 8

                observation[i].append(connection_feature)

        return np.array(observation)

    def observe_old(self):
        """
        Observation function calculated based on the traffic movements, each traffic movement can be described as:
        1. The length of incoming lane and outgoing lane
        2. The number of vehicles in the incoming lane and outgoing lane
        3. The number of halting vehicles in the incoming lane and outgoing lane
        """
        observation = []
        for movement in self.traffic_movement_list:
            in_lane, out_lane = movement

            length_in = traci.lane.getLength(in_lane) / 200
            length_out = traci.lane.getLength(out_lane) / 200
            num_veh_in = traci.lane.getLastStepVehicleNumber(in_lane) / 24
            num_veh_out = traci.lane.getLastStepVehicleNumber(out_lane) / 24
            num_halting_veh_in = np.clip(traci.lane.getLastStepHaltingNumber(in_lane), 0, 8) / 10
            num_halting_veh_out = np.clip(traci.lane.getLastStepHaltingNumber(out_lane), 0, 8) / 10

            observation.append([length_in, length_out, num_veh_in, num_veh_out, num_halting_veh_in, num_halting_veh_out])

        return np.array(observation)

    def observe(self, max_distance=None):
        """
        Observation function calculated based on the traffic movements, where each movement state can be defined as:
        1. The phase signal/activation status (0 for red signal and 1 for green signal)
        2. The number of stopped vehicles in the incoming lane and connected outgoing lane
        3. The number of moving vehicles in the incoming lane and connected outgoing lane
        """
        observation = []
        real_phase_code = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        phase_code = self.valid_phase_list[self.action_space.index(real_phase_code)]
        for i, movement in enumerate(self.traffic_movement_list):
            in_lane, out_lane = movement

            # The activation status of the traffic movement
            if phase_code[i] == 'G':
                phase_signal = 1
            elif phase_code[i] == 'g':
                phase_signal = 0.5
            else:
                phase_signal = 0

            # For most cases, there is no replaced lane dict
            if len(self.replaced_lane_dict) == 0:
                if max_distance is None:
                    num_veh_stopped_in = traci.lane.getLastStepHaltingNumber(in_lane)
                    num_veh_stopped_out = traci.lane.getLastStepHaltingNumber(out_lane)
                    num_veh_moving_in = (traci.lane.getLastStepVehicleNumber(in_lane) - traci.lane.getLastStepHaltingNumber(in_lane))
                    num_veh_moving_out = (traci.lane.getLastStepVehicleNumber(out_lane) - traci.lane.getLastStepHaltingNumber(out_lane))

                else:
                    # Calculate the count of moving vehicles and stopped vehicles on the incoming lane
                    num_veh_moving_in, num_veh_stopped_in = 0, 0
                    in_lane_veh_list = traci.lane.getLastStepVehicleIDs(in_lane)
                    in_lane_Length = self.lane_length_dict[in_lane]
                    for v in in_lane_veh_list:
                        veh_pos = traci.vehicle.getLanePosition(v)
                        if (in_lane_Length - veh_pos) <= max_distance:
                            # vehicle speed smaller than 0.2 m/s will be considered as stopped
                            if traci.vehicle.getSpeed(v) < 0.2:
                                num_veh_stopped_in += 1
                            else:
                                num_veh_moving_in += 1

                    # Calculate the count of moving vehicles and stopped vehicles on the outgoing lane
                    num_veh_moving_out, num_veh_stopped_out = 0, 0
                    out_lane_veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                    out_lane_length = self.lane_length_dict[out_lane]
                    for v in out_lane_veh_list:
                        veh_pos = traci.vehicle.getLanePosition(v)
                        if (out_lane_length - veh_pos) <= max_distance:
                            if traci.vehicle.getSpeed(v) < 0.2:
                                num_veh_stopped_out += 1
                            else:
                                num_veh_moving_out += 1

            # The replaced lane dict only activated for Ingolstadt network to rectify controlled lanes.
            else:
                if max_distance is None:
                    if in_lane not in self.replaced_lane_dict:
                        num_veh_stopped_in = traci.lane.getLastStepHaltingNumber(in_lane)
                        num_veh_moving_in = (traci.lane.getLastStepVehicleNumber(in_lane) - traci.lane.getLastStepHaltingNumber(in_lane))
                    else:
                        num_veh_stopped_in, num_veh_moving_in = 0, 0
                        for replaced_lane in self.replaced_lane_dict[in_lane]:
                            num_veh_stopped_in += traci.lane.getLastStepHaltingNumber(replaced_lane)
                            num_veh_moving_in += (traci.lane.getLastStepVehicleNumber(replaced_lane) - traci.lane.getLastStepHaltingNumber(replaced_lane))

                    if out_lane not in self.replaced_lane_dict:
                        num_veh_stopped_out = traci.lane.getLastStepHaltingNumber(out_lane)
                        num_veh_moving_out = (traci.lane.getLastStepVehicleNumber(out_lane) - traci.lane.getLastStepHaltingNumber(out_lane))

                    else:
                        num_veh_stopped_out, num_veh_moving_out = 0, 0
                        for replaced_lane in self.replaced_lane_dict[out_lane]:
                            num_veh_stopped_out += traci.lane.getLastStepHaltingNumber(replaced_lane)
                            num_veh_moving_out += (traci.lane.getLastStepVehicleNumber(replaced_lane) - traci.lane.getLastStepHaltingNumber(replaced_lane))

                else:
                    num_veh_stopped_in, num_veh_moving_in = 0, 0
                    if in_lane not in self.replaced_lane_dict:
                        # Need to check the vehicle distance
                        in_lane_veh_list = traci.lane.getLastStepVehicleIDs(in_lane)
                        in_lane_Length = self.lane_length_dict[in_lane]
                        for v in in_lane_veh_list:
                            veh_pos = traci.vehicle.getLanePosition(v)
                            if (in_lane_Length - veh_pos) <= max_distance:
                                # vehicle speed smaller than 0.2 m/s will be considered as stopped
                                if traci.vehicle.getSpeed(v) < 0.2:
                                    num_veh_stopped_in += 1
                                else:
                                    num_veh_moving_in += 1
                    else:
                        # All replaced lanes are shorter than 200m, do not need to check distance here
                        for replaced_lane in self.replaced_lane_dict[in_lane]:
                            num_veh_stopped_in += traci.lane.getLastStepHaltingNumber(replaced_lane)
                            num_veh_moving_in += (traci.lane.getLastStepVehicleNumber(replaced_lane) - traci.lane.getLastStepHaltingNumber(replaced_lane))

                    num_veh_stopped_out, num_veh_moving_out = 0, 0
                    if out_lane not in self.replaced_lane_dict:
                        out_lane_veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                        out_lane_length = self.lane_length_dict[out_lane]
                        for v in out_lane_veh_list:
                            veh_pos = traci.vehicle.getLanePosition(v)
                            if (out_lane_length - veh_pos) <= max_distance:
                                if traci.vehicle.getSpeed(v) < 0.2:
                                    num_veh_stopped_out += 1
                                else:
                                    num_veh_moving_out += 1

                    else:
                        for replaced_lane in self.replaced_lane_dict[out_lane]:
                            num_veh_stopped_out += traci.lane.getLastStepHaltingNumber(replaced_lane)
                            num_veh_moving_out += (traci.lane.getLastStepVehicleNumber(replaced_lane) - traci.lane.getLastStepHaltingNumber(replaced_lane))

            # State normalization
            num_veh_moving_in /= 20
            num_veh_stopped_in /= 20
            num_veh_moving_out /= 20
            num_veh_stopped_out /= 20

            observation.append([phase_signal, num_veh_stopped_in, num_veh_stopped_out, num_veh_moving_in, num_veh_moving_out])

        return np.array(observation)

    def observe_ma2c_network(self):
        """
        The observation function specifically used for MA2C datasets (Grid Manhattan network and Monaco network),
        and GESA datasets (Shenzhen network and Shaoxing network), where the observation/state vectors are collected
        from the installed lane-area detectors.
        The observation for each traffic movement contains (with dimension 8):
        1. The phase signal/activation status (0 for red signal and 1 for green signal)
        2. The number of stopped vehicles in the incoming lane and connected outgoing lane
        3. The number of moving vehicles in the incoming lane and connected outgoing lane
        4. The occupancy (the percentage of space the detector was occupied by a vehicle [%]) in the incoming lane and outgoing lane
        5. The state of the outgoing lane (0/not controlled by traffic lights, 1/controlled by traffic lights)
        """
        all_detector_list = traci.lanearea.getIDList()

        observation = []
        real_phase_code = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        phase_code = self.valid_phase_list[self.action_space.index(real_phase_code)]
        for i, movement in enumerate(self.traffic_movement_list):
            in_lane, out_lane = movement
            in_detector = self.incoming_detector_list[self.incoming_lane_list.index(in_lane)]
            out_detector = self.outgoing_detector_list[self.outgoing_lane_list.index(out_lane)]

            # The activation status of the traffic movement
            if phase_code[i] == 'G':
                phase_signal = 1
            elif phase_code[i] == 'g':
                phase_signal = 0.5
            else:
                phase_signal = 0

            # Calculate the count of vehicles on the incoming lane-area detectors
            num_veh_stopped_in = traci.lanearea.getLastStepHaltingNumber(in_detector)
            num_veh_moving_in = (traci.lanearea.getLastStepVehicleNumber(in_detector) - num_veh_stopped_in)
            num_veh_occ_in = traci.lanearea.getLastStepOccupancy(in_detector) / 100

            # Calculate the count of vehicles on the outgoing lane-area detectors
            if out_detector in all_detector_list:
                num_veh_stopped_out = traci.lanearea.getLastStepHaltingNumber(out_detector)
                num_veh_moving_out = (traci.lanearea.getLastStepVehicleNumber(out_detector) - num_veh_stopped_out)
                num_veh_occ_out = traci.lanearea.getLastStepOccupancy(out_detector) / 100
                out_lane_state = 1

            else:
                # For ma2c dataset, the default detection range is 50m
                distance = 50
                out_lane_length = self.lane_length_dict[out_lane]
                out_lane_veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                num_veh_stopped_out, num_veh_moving_out, num_veh_occ_out, lane_veh_length = 0, 0, 0, 0
                for v in out_lane_veh_list:
                    veh_speed = traci.vehicle.getSpeed(v)
                    veh_length = traci.vehicle.getLength(v)
                    veh_pos = traci.vehicle.getLanePosition(v)
                    if (out_lane_length - veh_pos) <= distance:
                        if veh_speed < 0.1:
                            num_veh_stopped_out += 1
                        else:
                            num_veh_moving_out += 1
                        lane_veh_length += veh_length

                num_veh_occ_out = lane_veh_length / distance
                out_lane_state = 0

            # State normalization
            num_veh_stopped_in /= 10
            num_veh_stopped_out /= 10
            num_veh_moving_in /= 10
            num_veh_moving_out /= 10

            observation.append([phase_signal, num_veh_stopped_in, num_veh_stopped_out, num_veh_moving_in, num_veh_moving_out, num_veh_occ_in, num_veh_occ_out, out_lane_state])

        return np.array(observation)

    def observe_resco_network(self, out_lane_state_dict):
        """
        The observation function specifically used for RESCO datasets (Cologne network and Ingolstadt network),
        where the state vectors are collected from the virtual lane-area detectors (200m).
        The observation for each traffic movement contains:
        1. The phase signal/activation status (0 for red signal and 1 for green signal)
        2. The number of stopped vehicles in the incoming lane and connected outgoing lane
        3. The number of moving vehicles in the incoming lane and connected outgoing lane
        4. The occupancy of the lane-area detectors of the incoming lane detector and outgoing lane detector
        5. The state of the outgoing lane (0/not controlled by traffic lights, 1/controlled by traffic lights)
        @param out_lane_state_dict: a dictionary that contains the state of the outgoing lanes, whether controlled by the traffic light or not
        """
        distance = 50

        observation = []
        real_phase_code = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        phase_code = self.valid_phase_list[self.action_space.index(real_phase_code)]
        for i, movement in enumerate(self.traffic_movement_list):
            in_lane, out_lane = movement

            # The activation status of the traffic movement
            if phase_code[i] == 'G':
                phase_signal = 1
            elif phase_code[i] == 'g':
                phase_signal = 0.5
            else:
                phase_signal = 0

            # For most cases, there is no replaced lane dict
            num_veh_stopped_in, num_veh_moving_in, veh_occ_in = 0, 0, 0
            if in_lane not in self.replaced_lane_dict:
                # Need to check the vehicle distance
                in_lane_veh_list = traci.lane.getLastStepVehicleIDs(in_lane)
                in_lane_Length = self.lane_length_dict[in_lane]
                for v in in_lane_veh_list:
                    veh_pos = traci.vehicle.getLanePosition(v)
                    veh_speed = traci.vehicle.getSpeed(v)
                    veh_length = traci.vehicle.getLength(v)
                    if (in_lane_Length - veh_pos) <= distance:
                        # vehicle speed smaller than 0.1 m/s will be considered as stopped
                        if veh_speed < 0.1:
                            num_veh_stopped_in += 1
                        else:
                            num_veh_moving_in += 1
                        veh_occ_in += veh_length
            else:
                # All replaced lanes are shorter than 200m, do not need to check distance here
                for replaced_lane in self.replaced_lane_dict[in_lane]:
                    num_veh_stopped_in += traci.lane.getLastStepHaltingNumber(replaced_lane)
                    num_veh_moving_in += (traci.lane.getLastStepVehicleNumber(replaced_lane) - traci.lane.getLastStepHaltingNumber(replaced_lane))
                    in_veh_list = traci.lane.getLastStepVehicleIDs(replaced_lane)
                    for v in in_veh_list:
                        veh_length = traci.vehicle.getLength(v)
                        veh_occ_in += veh_length

            num_veh_stopped_out, num_veh_moving_out, veh_occ_out = 0, 0, 0
            if out_lane not in self.replaced_lane_dict:
                out_lane_veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                out_lane_length = self.lane_length_dict[out_lane]
                for v in out_lane_veh_list:
                    veh_pos = traci.vehicle.getLanePosition(v)
                    veh_speed = traci.vehicle.getSpeed(v)
                    veh_length = traci.vehicle.getLength(v)
                    if (out_lane_length - veh_pos) <= distance:
                        if veh_speed < 0.1:
                            num_veh_stopped_out += 1
                        else:
                            num_veh_moving_out += 1
                        veh_occ_out += veh_length
            else:
                for replaced_lane in self.replaced_lane_dict[out_lane]:
                    num_veh_stopped_out += traci.lane.getLastStepHaltingNumber(replaced_lane)
                    num_veh_moving_out += (traci.lane.getLastStepVehicleNumber(replaced_lane) - traci.lane.getLastStepHaltingNumber(replaced_lane))
                    out_veh_list = traci.lane.getLastStepVehicleIDs(replaced_lane)
                    for v in out_veh_list:
                        veh_length = traci.vehicle.getLength(v)
                        veh_occ_out += veh_length

            # State normalization
            num_veh_moving_in /= 20
            num_veh_stopped_in /= 20
            num_veh_moving_out /= 20
            num_veh_stopped_out /= 20
            veh_occ_in /= distance
            veh_occ_out /= distance
            out_lane_state = out_lane_state_dict[out_lane]

            observation.append([phase_signal, num_veh_stopped_in, num_veh_stopped_out, num_veh_moving_in, num_veh_moving_out, veh_occ_in, veh_occ_out, out_lane_state])

        return np.array(observation)

    def observe_gesa_network(self):
        """
        The observation function specifically used for GESA-based methods, where the observation/state vectors are
        collected from the installed lane-area detectors.
        The observation for each traffic movement contains (with dimension 12):
        1. The phase signal/activation status (0 for red signal and 1 for green signal)
        2. The number of stopped vehicles in the incoming lane and connected outgoing lane
        3. The number of moving vehicles in the incoming lane and connected outgoing lane
        4. The queue length (in meters) in the incoming lane and connected outgoing lane
        5. The incoming flow rate (veh/s) in the incoming lane and connected outgoing lane
        6. The occupancy (the percentage of space the detector was occupied by a vehicle [%]) in the incoming lane and outgoing lane
        7. The state of the outgoing lane (0/not controlled by traffic lights, 1/controlled by traffic lights)
        """
        all_detector_list = traci.lanearea.getIDList()

        observation = []
        real_phase_code = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        phase_code = self.valid_phase_list[self.action_space.index(real_phase_code)]
        flow_rate_in_lane, flow_rate_out_lane = self.get_lane_flow_rate()
        for i, movement in enumerate(self.traffic_movement_list):
            in_lane, out_lane = movement
            in_detector = self.incoming_detector_list[self.incoming_lane_list.index(in_lane)]
            out_detector = self.outgoing_detector_list[self.outgoing_lane_list.index(out_lane)]

            # The activation status of the traffic movement
            if phase_code[i] == 'G':
                phase_signal = 1
            elif phase_code[i] == 'g':
                phase_signal = 0.5
            else:
                phase_signal = 0

            # Calculate the count of vehicles on the incoming lane-area detectors
            flow_rate_in = flow_rate_in_lane[in_lane]
            queue_length_in = traci.lanearea.getJamLengthMeters(in_detector)
            num_veh_stopped_in = traci.lanearea.getLastStepHaltingNumber(in_detector)
            num_veh_moving_in = (traci.lanearea.getLastStepVehicleNumber(in_detector) - num_veh_stopped_in)
            num_veh_occ_in = traci.lanearea.getLastStepOccupancy(in_detector) / 100

            # Calculate the count of vehicles on the outgoing lane-area detectors
            flow_rate_out = flow_rate_out_lane[out_lane]
            if out_detector in all_detector_list:
                queue_length_out  = traci.lanearea.getJamLengthMeters(out_detector)
                num_veh_stopped_out = traci.lanearea.getLastStepHaltingNumber(out_detector)
                num_veh_moving_out = (traci.lanearea.getLastStepVehicleNumber(out_detector) - num_veh_stopped_out)
                num_veh_occ_out = traci.lanearea.getLastStepOccupancy(out_detector) / 100
                out_lane_state = 1

            else:
                # The default detection range is 50m
                distance = 50
                out_lane_length = self.lane_length_dict[out_lane]
                out_lane_veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                num_veh_stopped_out, num_veh_moving_out, num_veh_occ_out, lane_veh_length = 0, 0, 0, 0
                for v in out_lane_veh_list:
                    veh_speed = traci.vehicle.getSpeed(v)
                    veh_length = traci.vehicle.getLength(v)
                    veh_pos = traci.vehicle.getLanePosition(v)
                    if (out_lane_length - veh_pos) <= distance:
                        if veh_speed < 0.1:
                            num_veh_stopped_out += 1
                        else:
                            num_veh_moving_out += 1
                        lane_veh_length += veh_length

                num_veh_occ_out = lane_veh_length / distance
                queue_length_out = -1
                out_lane_state = 0

            # State normalization
            num_veh_stopped_in /= 10
            num_veh_stopped_out /= 10
            num_veh_moving_in /= 10
            num_veh_moving_out /= 10
            queue_length_in /= 100
            queue_length_out /= 100
            flow_rate_in /= 3
            flow_rate_out /= 3

            observation.append([phase_signal, num_veh_stopped_in, num_veh_stopped_out,
                                num_veh_moving_in, num_veh_moving_out, num_veh_occ_in, num_veh_occ_out,
                                queue_length_in, queue_length_out, flow_rate_in, flow_rate_out, out_lane_state])

        return np.array(observation)

    def observe_AttendLight(self, max_distance=None):
        """
        Observation function calculated based on the participating lanes of each phase (original paper definition):
        1. The number of halting vehicles
        2. The number of moving vehicles
        """
        links_incoming = []
        links_outgoing = []
        for k, v in self.lane_links.items():
            links_incoming.extend([k] * len(v))
            links_outgoing.extend(v)

        obs = []
        for i, phase_code in enumerate(self.valid_phase_list):
            parti_in_lane_set = set()
            parti_out_lane_set = set()

            # Collect participating lanes based on phase code
            for j, code in enumerate(phase_code):
                if code.upper() == 'G':
                    parti_in_lane_set.add(links_incoming[j])
                    parti_out_lane_set.add(links_outgoing[j])

            parti_lane_list = list(parti_in_lane_set | parti_out_lane_set)

            phase_obs = []
            # Calculate observations based on lane data
            for lane in parti_lane_list:
                if max_distance is None:
                    phase_obs.append([
                        traci.lane.getLastStepVehicleNumber(lane) / 24,
                        traci.lane.getLastStepHaltingNumber(lane) / 8
                    ])
                else:
                    lane_length = self.lane_length_dict[lane]
                    veh_list = traci.lane.getLastStepVehicleIDs(lane)
                    num_veh, num_stopped_veh = 0, 0
                    for v in veh_list:
                        veh_pos = traci.vehicle.getLanePosition(v)
                        if (lane_length - veh_pos) <= max_distance:
                            num_veh += 1
                            if traci.vehicle.getSpeed(v) < 0.2:
                                num_stopped_veh += 1
                    phase_obs.append([num_veh / 24, num_stopped_veh / 8])

            obs.append(phase_obs)

        return obs

    def observe_AttendLight_ma2c_network(self):
        """
        Observation function calculated based on the lane-area detectors of participate lanes of each phase:
        1. The number of halting vehicles
        2. The number of moving vehicles
        """
        all_detector_list = traci.lanearea.getIDList()

        links_incoming = []
        links_outgoing = []
        for k, v in self.lane_links.items():
            links_incoming.extend([k] * len(v))
            links_outgoing.extend(v)

        obs = []
        for i, phase_code in enumerate(self.valid_phase_list):
            parti_in_lane_set = set()
            parti_out_lane_set = set()

            # Collect participating lanes
            for j, code in enumerate(phase_code):
                if code.upper() == 'G':  # Simplify condition, handle both 'G' and 'g'
                    parti_in_lane_set.add(links_incoming[j])
                    parti_out_lane_set.add(links_outgoing[j])

            # Mapping lanes to detectors
            in_lane_to_detector = {lane: self.incoming_detector_list[self.incoming_lane_list.index(lane)] for lane in
                                   parti_in_lane_set}
            out_lane_to_detector = {lane: self.outgoing_detector_list[self.outgoing_lane_list.index(lane)] for lane in
                                    parti_out_lane_set if self.outgoing_detector_list[self.outgoing_lane_list.index(lane)]
                                    in all_detector_list}

            # Combine participating detectors and collect observations
            phase_obs = [
                [traci.lanearea.getLastStepVehicleNumber(detector) / 10,
                 traci.lanearea.getLastStepHaltingNumber(detector) / 10,
                 traci.lanearea.getLastStepOccupancy(detector) / 100]
                for detector in list(in_lane_to_detector.values()) + list(out_lane_to_detector.values())
            ]

            obs.append(phase_obs)

        return obs

    def observe_AttendLight_resco_network(self):
        """
        Observation function calculated based on the lane-area detectors of participate lanes of each phase:
        1. The number of halting vehicles
        2. The number of moving vehicles
        3. The occupancy within a specified distance
        If no detector is available on the outgoing lanes, the data can be substituted with lane attributes.
        """
        distance = 50
        all_detector_list = traci.lanearea.getIDList()

        links_incoming = []
        links_outgoing = []
        for k, v in self.lane_links.items():
            links_incoming.extend([k] * len(v))
            links_outgoing.extend(v)

        obs = []
        for i, phase_code in enumerate(self.valid_phase_list):
            parti_in_lane_set = set()
            parti_out_lane_set = set()

            # Collect participating lanes
            for j, code in enumerate(phase_code):
                if code.upper() == 'G':  # Simplify condition, handle both 'G' and 'g'
                    parti_in_lane_set.add(links_incoming[j])
                    parti_out_lane_set.add(links_outgoing[j])

            # Mapping lanes to detectors
            in_lane_to_detector = {lane: self.incoming_detector_list[self.incoming_lane_list.index(lane)] for lane in
                                   parti_in_lane_set}
            out_lane_to_detector = {lane: self.outgoing_detector_list[self.outgoing_lane_list.index(lane)] for lane in
                                    parti_out_lane_set if self.outgoing_detector_list[self.outgoing_lane_list.index(lane)]
                                    in all_detector_list}

            # Check whether incoming detectors and outgoing detectors are in the detector list
            phase_obs = []
            for detector in (list(in_lane_to_detector.values()) + list(out_lane_to_detector.values())):
                if detector in all_detector_list:
                    veh_occ = traci.lanearea.getLastStepOccupancy(detector) / 100
                    num_veh_stopped = traci.lanearea.getLastStepHaltingNumber(detector) / 10
                    num_veh_moving = (traci.lanearea.getLastStepVehicleNumber(detector) - num_veh_stopped) / 10
                else:
                    num_veh_stopped, num_veh_moving, veh_occ = 0, 0, 0
                    lane_length = self.lane_length_dict[detector]
                    lane_veh_list = traci.lane.getLastStepVehicleIDs(detector)
                    for v in lane_veh_list:
                        veh_pos = traci.vehicle.getLanePosition(v)
                        veh_speed = traci.vehicle.getSpeed(v)
                        veh_length = traci.vehicle.getLength(v)
                        if (lane_length - veh_pos) <= distance:
                            if veh_speed < 0.1:
                                num_veh_stopped += 1
                            else:
                                num_veh_moving += 1
                            veh_occ += veh_length
                        else:
                            pass

                    num_veh_stopped /= 10
                    num_veh_moving /= 10
                    veh_occ /= distance

                phase_obs.append([num_veh_stopped, num_veh_moving, veh_occ])

            obs.append(phase_obs)

        return obs

    def calculate_target_queue(self, norm=True, lane=False):
        """
        Calculate the target/ground of truth for queue prediction
        @param norm: bool -> value normalized
        @param lane: bool -> values from lane or detector
        @return: numpy array, target queue value
        """
        target_queue = np.zeros(len(self.incoming_lane_list))
        for i, (in_lane, detector) in enumerate(zip(self.incoming_lane_list, self.incoming_detector_list)):
            if lane:
                target_queue[i] = traci.lane.getLastStepHaltingNumber(in_lane) / 24 \
                    if norm else traci.lane.getLastStepHaltingNumber(in_lane)
            else:
                target_queue[i] = traci.lanearea.getLastStepHaltingNumber(detector) / 7 \
                    if norm else traci.lanearea.getLastStepHaltingNumber(detector)

        return target_queue.tolist()

    def get_random_action(self):
        """
        Choose the action randomly
        """
        action = np.random.randint(0, self.action_space_n)

        return action

    def get_fixed_time_action(self):
        """
        Choose the phase that follows the pre-defined order
        """
        curr_phase_code = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        next_action = (self.action_space.index(curr_phase_code) + 1) % self.action_space_n

        return next_action

    def get_greedy_action(self, detector=True):
        """
        Choose the phase that can move the maximum number of halted vehicles
        @return:
        """
        r = np.zeros(self.action_space_n)
        for i, phase in enumerate(self.valid_phase_list):
            for j, code in enumerate(phase):
                in_lane, out_lane = self.traffic_movement_list[j]
                if code.lower() == 'g':
                    if detector:
                        in_detector = self.lane_detector_dict[in_lane]
                        r[i] += traci.lanearea.getLastStepHaltingNumber(in_detector)
                    else:
                        r[i] += traci.lane.getLastStepHaltingNumber(in_lane)

        return np.argmax(r)

    def get_pressure_action(self, detector=True):
        """
        Choose the phase that can reduce the most phase pressure
        """
        all_detector_list = traci.lanearea.getIDList()

        r = np.zeros(self.action_space_n)
        for i, phase in enumerate(self.valid_phase_list):
            for j, code in enumerate(phase):
                if code.lower() == 'g':
                    in_lane, out_lane = self.traffic_movement_list[j]
                    if detector:
                        in_detector = self.lane_detector_dict[in_lane]
                        out_detector = self.lane_detector_dict[out_lane]
                        if out_detector in all_detector_list:
                            r[i] += (traci.lanearea.getLastStepHaltingNumber(in_detector) -
                                     traci.lanearea.getLastStepHaltingNumber(out_detector))
                        else:
                            r[i] += (traci.lanearea.getLastStepHaltingNumber(in_detector) -
                                     traci.lane.getLastStepHaltingNumber(out_lane))
                    else:
                        r[i] += (traci.lane.getLastStepHaltingNumber(in_lane) -
                                 traci.lane.getLastStepVehicleNumber(out_lane))

        return np.argmax(r)

    def get_truncated_queue_reward(self, detector=False, queue_range=50):
        """"
        Calculate reward based on the queue of lane area detectors
        """
        queue = []
        # if there are lane area detectors
        if detector:
            for i, detector in enumerate(self.incoming_detector_list):
                queue.append(traci.lanearea.getLastStepHaltingNumber(detector) / 7)
        else:
            for i, in_lane in enumerate(self.incoming_lane_list):
                if in_lane not in self.replaced_lane_dict:
                    num_stopped_veh = 0
                    lane_length = traci.lane.getLength(in_lane)
                    veh_list = traci.lane.getLastStepVehicleIDs(in_lane)
                    for v in veh_list:
                        veh_pos = traci.vehicle.getLanePosition(v)
                        veh_speed = traci.vehicle.getSpeed(v)
                        # Calculate the queue length for each incoming lane, if and only if the speed of the vehicle is
                        # lower than 0.2 m/s and the position is within the range of 50 meter near the intersection.
                        if veh_pos >= (lane_length - queue_range) and veh_speed <= 0.2:
                            num_stopped_veh += 1
                        else:
                            pass
                else:
                    num_stopped_veh = 0
                    # If the replaced lane is shorter than queue length detection range
                    if self.replaced_lane_length_dict[in_lane] <= queue_range:
                        for replaced_lane in self.replaced_lane_dict[in_lane]:
                            veh_list = traci.lane.getLastStepVehicleIDs(replaced_lane)
                            for v in veh_list:
                                if traci.vehicle.getSpeed(v) < 0.2:
                                    num_stopped_veh += 1
                                else:
                                    pass
                    # If the replaced lane is larger than the queue length detection range, take the second lane segment,
                    # since the second lane segment is longer.
                    else:
                        lane_length = traci.lane.getLength(self.replaced_lane_dict[in_lane][-1])
                        veh_list = traci.lane.getLastStepVehicleIDs(self.replaced_lane_dict[in_lane][-1])
                        for v in veh_list:
                            veh_pos = traci.vehicle.getLanePosition(v)
                            veh_speed = traci.vehicle.getSpeed(v)
                            if veh_pos >= (lane_length - queue_range) and veh_speed <= 0.2:
                                num_stopped_veh += 1
                            else:
                                pass

                queue.append(num_stopped_veh / 7)
        # reward = -1 * np.sum(np.array(queue)) / 10
        reward = -1 * np.mean(np.array(queue))
        return reward

    def get_truncated_queue_reward_ma2c_network(self, regional_reward=False):
        """
        The reward function specifically used for MA2C datasets (Grid Manhattan network and Monaco network),
        where the reward is calculated by the queue length on the lane-area detectors with 50 meters.
        @params regional_reward: whether adding the queue length reward on the outgoing lanes for coordination
        """
        if not regional_reward:
            queue_reward = []
            for i, in_detector in enumerate(self.incoming_detector_list):
                queue_reward.append(traci.lanearea.getLastStepHaltingNumber(in_detector) / 7)
            reward = -1 * np.mean(np.array(queue_reward))
        else:
            distance = 50
            in_queue_reward = []
            for i, in_detector in enumerate(self.incoming_detector_list):
                in_queue_reward.append(traci.lanearea.getLastStepHaltingNumber(in_detector) / 7)

            out_queue_reward = []
            all_detector_list = traci.lanearea.getIDList()
            for j, out_detector in enumerate(self.outgoing_detector_list):
                if out_detector in all_detector_list:
                    out_queue_reward.append(traci.lanearea.getLastStepHaltingNumber(out_detector) / 7)
                else:
                    queue = 0
                    out_lane = self.outgoing_lane_list[j]
                    out_lane_length = self.lane_length_dict[out_lane]
                    veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                    for v in veh_list:
                        veh_speed = traci.vehicle.getSpeed(v)
                        veh_pos = traci.vehicle.getLanePosition(v)
                        if (out_lane_length - veh_pos) <= distance:
                            if veh_speed < 0.2:
                                queue += 1
                    out_queue_reward.append(queue / 7)

            reward = -1 * (np.mean(np.array(in_queue_reward)) + np.mean(np.array(out_queue_reward)))
        return reward

    def get_truncated_queue_reward_resco_network(self, distance=50, regional_reward=False):
        """
        The reward function specifically used for MA2C datasets (Grid Manhattan network and Monaco network),
        where the reward is calculated by the queue length on the lane-area detectors with 50 meters.
        @params regional_reward: whether adding the queue length reward on the outgoing lanes for coordination
        """
        queue_reward_in = []
        for i, in_lane in enumerate(self.incoming_lane_list):
            if in_lane not in self.replaced_lane_dict:
                queue = 0
                in_lane_length = self.lane_length_dict[in_lane]
                in_lane_veh_list = traci.lane.getLastStepVehicleIDs(in_lane)
                for v in in_lane_veh_list:
                    veh_pos = traci.vehicle.getLanePosition(v)
                    veh_speed = traci.vehicle.getSpeed(v)
                    if (in_lane_length - veh_pos) <= distance:
                        if veh_speed < 0.2:
                            queue += 1
                queue_reward_in.append(queue)

            else:
                queue = 0
                for replaced_lane in self.replaced_lane_dict[in_lane]:
                    in_lane_veh_list = traci.lane.getLastStepVehicleIDs(replaced_lane)
                    for v in in_lane_veh_list:
                        veh_speed = traci.vehicle.getSpeed(v)
                        if veh_speed < 0.2:
                            queue += 1
                queue_reward_in.append(queue)

        if not regional_reward:
            reward = -1 * np.mean(queue_reward_in) / 10

        else:
            queue_reward_out = []
            for j, out_lane in enumerate(self.outgoing_lane_list):
                queue = 0
                out_lane_length = self.lane_length_dict[out_lane]
                out_lane_veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
                for v in out_lane_veh_list:
                    veh_pos = traci.vehicle.getLanePosition(v)
                    veh_speed = traci.vehicle.getSpeed(v)
                    if (out_lane_length - veh_pos) <= distance and veh_speed < 0.2:
                        queue += 1
                queue_reward_out.append(queue)

            reward = -1 * (np.mean(queue_reward_in) + np.mean(queue_reward_out)) / 10

        return reward

    def get_mixed_reward(self, queue_norm=7, wait_norm=300, queue_range=50):
        """
        Calculate the reward based on the combination of queue length and the waiting time of the first vehicle on each
        incoming lane to ensure the fairness
        """
        queue, wait = [], []
        for i, lane in enumerate(self.incoming_lane_list):
            num_stopped_veh = 0
            lane_length = traci.lane.getLength(lane)
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            # Calculate the first vehicle at each incoming lane
            front_v = None
            front_pos = None
            for v in veh_list:
                veh_speed = traci.vehicle.getSpeed(v)
                veh_pos = traci.vehicle.getLanePosition(v)
                if front_v is None:
                    front_v = v
                    front_pos = veh_pos
                else:
                    if veh_pos > front_pos:
                        front_v = v
                        front_pos = veh_pos

                # Calculate the queue length for each incoming lane, if and only if the speed of the vehicle is
                # lower than 0.2 m/s and the position is within the range of 50 meter near the intersection.
                if veh_pos >= (lane_length - queue_range) and veh_speed <= 0.2:
                    num_stopped_veh += 1
                else:
                    pass

            # Calculate the truncated queue length for each incoming lane
            queue.append(np.clip(num_stopped_veh / queue_norm, 0, 1))

            # Calculate waiting time for the first vehicle (near the intersection) at each incoming lane
            if front_v is not None and front_pos >= (lane_length - 15):
                wait.append(np.clip(traci.vehicle.getWaitingTime(front_v) / wait_norm, 0, 1))
            else:
                wait.append(0)
        reward = -1 * (np.sum(queue) + np.sum(wait)) / 15
        return reward

    def get_regional_queue_reward(self, distance=50, out_queue_weight=1):
        """"
        Calculate regional reward based on the queue length of the incoming lane outgoing lanes at the intersection
        """
        in_queue_list = []
        out_queue_list = []
        for in_lane in self.incoming_lane_list:
            queue = 0
            lane_length = self.lane_length_dict[in_lane]
            veh_list = traci.lane.getLastStepVehicleIDs(in_lane)
            for v in veh_list:
                veh_pos = traci.vehicle.getLanePosition(v)
                if veh_pos > (lane_length - distance) and traci.vehicle.getSpeed(v) < 0.2:
                    queue += 1
                else:
                    pass
            in_queue_list.append(queue)

        for out_lane in self.outgoing_lane_list:
            queue = 0
            lane_length = self.lane_length_dict[out_lane]
            veh_list = traci.lane.getLastStepVehicleIDs(out_lane)
            for v in veh_list:
                veh_pos = traci.vehicle.getLanePosition(v)
                if veh_pos > (lane_length - distance) and traci.vehicle.getSpeed(v) < 0.2:
                    queue += 1
                else:
                    pass
            out_queue_list.append(queue)

        queue_reward = np.mean(in_queue_list) + out_queue_weight * np.mean(out_queue_list)
        return queue_reward

    def get_lane_flow_rate(self, interval=15):
        """
        Calculate in flow rate (veh/s) for both incoming lanes and outgoing lanes for GESA-based observation function
        """
        flow_rate_in_lane = dict()
        flow_rate_out_lane = dict()
        curr_veh_list_in_lane = dict()
        curr_veh_list_out_lane = dict()
        for i, in_lane in self.incoming_lane_list:
            curr_veh_list_in_lane[in_lane] = traci.lane.getLastStepVehicleIDs(in_lane)
            in_veh_in_lane = set(curr_veh_list_in_lane[in_lane]) - set(self.pre_veh_list_in_lane[in_lane])
            flow_rate_in_lane[in_lane] = len(in_veh_in_lane) / interval

        for i, out_lane in self.outgoing_lane_list:
            curr_veh_list_out_lane[out_lane] = traci.lane.getLastStepVehicleIDs(out_lane)
            in_veh_out_lane = set(curr_veh_list_out_lane[out_lane]) - set(self.pre_veh_list_out_lane[out_lane])
            flow_rate_out_lane[out_lane] = len(in_veh_out_lane) / interval

        self.pre_veh_list_in_lane = curr_veh_list_in_lane
        self.pre_veh_list_out_lane = curr_veh_list_out_lane

        return flow_rate_in_lane, flow_rate_out_lane
