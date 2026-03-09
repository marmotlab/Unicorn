import traci
import numpy as np

from sumolib import checkBinary
from utils import save_as_json
from collections import OrderedDict


class net_config_saver:
    def __init__(self):
        self.incoming_lane_list = None
        self.outgoing_lane_list = None
        self.incoming_detector_list = None
        self.outgoing_detector_list = None
        self.lane_links = None
        self.num_links_lane = None
        self.tlc_state = None
        self.action_space = None
        self.neighbor_map = None
        self.action_space_n = None
        self.phase_list = None
        self.valid_phase_list = None

    @staticmethod
    def start_simulation(sumo_config_file_path):
        sumoBinary = checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", sumo_config_file_path,
                     "--start",
                     "--quit-on-end"])

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))

        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()

        return self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list, self.lane_links, list(self.num_links_lane)

    def save_net_config(self, config_data_save_path):
        """
        save net configuration data
        :param config_data_save_path
        :return: a dictionary contains the net configuration data, which contains the following items
        'incoming_lane_list': a list contains the ids of the incoming lanes
        'outgoing_lane_list':  a list contains the ids of the outgoing lanes
        'detector_list': a list contains the ids of the detectors
        'lane_links': a dictionary (key/id of the incoming lane, value/ids of the connected outgoing lanes)
        'num_links_lane': np.array contains the number of connections for each incoming lane
        'tlc_state': a list contains the activated incoming lanes for each corresponding phase (1/Move, 0/Stop)
        'action_space': a list contains the GreenYellowRed codes for phases
        'neighbor_map': a list contains the ids of the neighbor intersections
        """
        net_config = {}
        np.save(config_data_save_path, net_config)


class grid_network_config_saver(net_config_saver):
    def __init__(self):
        super(grid_network_config_saver, self).__init__()

    @staticmethod
    def get_neighbor_map(padding=True):
        """
        collect the neighbor ids of target intersection
        :params: padding: the id of non-existed neighbor will be None to achieve four neighbors
        :return: a dictionary, return the target intersection as key, neighbor intersections as value
        """
        neighbour_dict = {}
        if padding:
            agent_id = []
            for i in range(1, 26):
                agent_id.append('nt{}'.format(str(i)))
            for x, y in enumerate(agent_id):
                neighbour_dict[y] = []
                # possible_neighbours = [-1, +1, -5, +5]
                possible_neighbours = [+5, +1, -5, -1]
                for element in possible_neighbours:
                    if 1 <= (x + 1 + element) <= 25:
                        if (x + 1) % 5 == 0 and element == 1:
                            neighbour_dict[y].append(None)
                        elif (x + 1) % 5 == 1 and element == -1:
                            neighbour_dict[y].append(None)
                        else:
                            neighbour_dict[y].append('nt{}'.format(x + element + 1))
                    else:
                        neighbour_dict[y].append(None)
        else:
            neighbor_dict = {'nt1': ['nt6', 'nt2'], 'nt5': ['nt10', 'nt4'], 'nt21': ['nt22', 'nt16'],
                             'nt25': ['nt20', 'nt24'], 'nt2': ['nt7', 'nt3', 'nt1'], 'nt3': ['nt8', 'nt4', 'nt2'],
                             'nt4': ['nt9', 'nt5', 'nt3'], 'nt22': ['nt23', 'nt17', 'nt21'],
                             'nt23': ['nt24', 'nt18', 'nt22'], 'nt24': ['nt25', 'nt19', 'nt23'],
                             'nt10': ['nt15', 'nt5', 'nt9'], 'nt15': ['nt20', 'nt10', 'nt14'],
                             'nt20': ['nt25', 'nt15', 'nt19'], 'nt6': ['nt11', 'nt7', 'nt1'],
                             'nt11': ['nt16', 'nt12', 'nt6'], 'nt16': ['nt21', 'nt17', 'nt11']}

            for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
                n_node = 'nt' + str(i + 5)
                s_node = 'nt' + str(i - 5)
                w_node = 'nt' + str(i - 1)
                e_node = 'nt' + str(i + 1)
                cur_node = 'nt' + str(i)
                neighbor_dict[cur_node] = [n_node, e_node, s_node, w_node]

        return neighbour_dict

    @staticmethod
    def get_phase_connected_lane_state(lane_links, phase_list, incoming_lane_list, outgoing_lane_list):
        """
        get the activated incoming lanes and outgoing lanes for each phase
        @param lane_links: dictionary that describes the connections between incoming lanes and outgoing lanes
        @param phase_list: list which contains phase code
        @param incoming_lane_list: list which contains ids of incoming lanes
        @param outgoing_lane_list: list which contains ids of outgoing lanes
        @return: dictionary, key/phase code, value/ two numpy 0/1 arrays, one indicates the activated incoming lanes
        (wrt. incoming lane list), the other one indicates the activated outgoing lanes (wrt. outgoing lane list).
        """
        lane_connections = {'in': [], 'out': []}
        for k, v in lane_links.items():
            for out_lane in v:
                lane_connections['in'].append(k)
                lane_connections['out'].append(out_lane)
        phase_lane_state = {}
        for phase in phase_list:
            state = np.zeros((len(incoming_lane_list), len(outgoing_lane_list)))
            for i, code in enumerate(phase):
                if code == 'G' or code == 'g':
                    # in_state = incoming_lane_list.index(lane_connections['in'][i])
                    # out_state = outgoing_lane_list.index(lane_connections['out'][i])
                    state[incoming_lane_list.index(lane_connections['in'][i])][
                        outgoing_lane_list.index(lane_connections['out'][i])] = 1
            phase_lane_state[phase] = state.tolist()

        return phase_lane_state

    def save_net_config(self, config_data_save_path='./grid_network_5_5/grid_network_5_5_config.json', save_file=False):
        """
        Save net configuration data to the json/numpy file
        """
        self.start_simulation(sumo_config_file_path='grid_network_5_5/grid_network_5_5.sumocfg')
        net_config = OrderedDict()
        neighbor_dict = self.get_neighbor_map(padding=True)
        for i in range(25):
            tls = 'nt{}'.format(i + 1)

            in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane = self.get_net_config(tls_id=tls)

            phase_list = ['GGgrrrGGgrrr',
                          'rrrGrGrrrGrG',
                          'rrrGGrrrrGGr',
                          'rrrGGGrrrrrr',
                          'rrrrrrrrrGGG']
            valid_phase_list = phase_list

            phase_type = str(len(phase_list))
            phase_type_index = i
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = traci.junction.getPosition(tls)

            neighbor_list = neighbor_dict[tls]

            phase_lane_state = self.get_phase_connected_lane_state(lane_links=lane_links,
                                                                   phase_list=phase_list,
                                                                   incoming_lane_list=in_lane_list,
                                                                   outgoing_lane_list=out_lane_list)

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in self.incoming_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)
            for out_lane in self.outgoing_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_list,
                'neighbor_list': neighbor_list,
                'phase_lane_state': phase_lane_state,
                'valid_phase_list': valid_phase_list,

                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
            }

        if save_file:
            suffix = config_data_save_path.split('.')[-1]
            if suffix == 'npy':
                np.save(config_data_save_path, net_config)
            else:
                save_as_json(config_data_save_path, net_config)

            print("Save Manhattan config data!")

        else:
            print('Not saving configurations!')


class monaco_network_config_saver(net_config_saver):
    def __init__(self):
        super().__init__()

        self.tls_data_dict = {'10026': ('6.0', ['9431', '9561', 'cluster_9563_9597', '9531']),
                              '8794': ('4.0', ['cluster_8985_9609', '9837', '9058', 'cluster_9563_9597']),
                              '8940': ('2.1', ['9007', '9429']),
                              '8996': ('2.2', ['cluster_9389_9689', '9713']),
                              '9007': ('2.3', ['9309', '8940']),
                              '9058': ('4.0', ['cluster_8985_9609', '8794', 'joinedS_0']),
                              '9153': ('2.0', ['9643']),
                              '9309': ('4.0', ['9466', '9007', 'cluster_9043_9052']),
                              '9413': ('2.3', ['9721', '9837']),
                              '9429': ('5.0', ['cluster_9043_9052', 'joinedS_1', '8940']),
                              '9431': ('2.4', ['9721', '9884', '9561', '10026']),
                              '9433': ('2.5', ['joinedS_1']),
                              '9466': ('4.0', ['9309', 'joinedS_0', 'cluster_9043_9052']),
                              '9480': ('2.3', ['8996', '9713']),
                              '9531': ('2.6', ['joinedS_1', '10026']),
                              '9561': ('4.0', ['cluster_9389_9689', '10026', '9431', '9884']),
                              '9643': ('2.3', ['9153']),
                              '9713': ('3.0', ['9721', '9884', '8996']),
                              '9721': ('6.0', ['9431', '9713', '9413']),
                              '9837': ('3.1', ['9413', '8794', 'cluster_8985_9609']),
                              '9884': ('2.7', ['9713', '9431', 'cluster_9389_9689', '9561']),
                              'cluster_8751_9630': ('4.0', ['cluster_9389_9689']),
                              'cluster_8985_9609': ('4.0', ['9837', '8794', '9058']),
                              'cluster_9043_9052': ('4.1', ['cluster_9563_9597', '9466', '9309', '10026', 'joinedS_1']),
                              'cluster_9389_9689': ('4.0', ['9884', '9561', 'cluster_8751_9630', '8996']),
                              'cluster_9563_9597': ('4.2', ['10026', '8794', 'joinedS_0', 'cluster_9043_9052']),
                              'joinedS_0': ('6.1', ['9058', 'cluster_9563_9597', '9466']),
                              'joinedS_1': ('3.2', ['9531', '9429'])}

        self.tls_phase_dict = {'4.0': ['GGgrrrGGgrrr', 'rrrGGgrrrGGg', 'rrGrrrrrGrrr', 'rrrrrGrrrrrG'],
                               '4.1': ['GGgrrGGGrrr', 'rrGrrrrrrrr', 'rrrGgrrrGGg', 'rrrrGrrrrrG'],
                               '4.2': ['GGGGrrrrrrrr', 'GGggrrGGggrr', 'rrrGGGGrrrrr', 'grrGGggrrGGg'],
                               '2.0': ['GGrrr', 'ggGGG'],
                               '2.1': ['GGGrrr', 'rrGGGg'],
                               '2.2': ['Grr', 'gGG'],
                               '2.3': ['GGGgrr', 'GrrrGG'],
                               '2.4': ['GGGGrr', 'rrrrGG'],
                               '2.5': ['Gg', 'rG'],
                               '2.6': ['GGGg', 'rrrG'],
                               '2.7': ['GGg', 'rrG'],
                               '3.0': ['GGgrrrGGg', 'rrGrrrrrG', 'rrrGGGGrr'],
                               '3.1': ['GgrrGG', 'rGrrrr', 'rrGGGr'],
                               '3.2': ['GGGGrrrGG', 'rrrrGGGGr', 'GGGGrrGGr'],
                               '5.0': ['GGGGgrrrrGGGggrrrr', 'grrrGrrrrgrrGGrrrr', 'GGGGGrrrrrrrrrrrrr',
                                       'rrrrrrrrrGGGGGrrrr', 'rrrrrGGggrrrrrggGg'],
                               '6.0': ['GGGgrrrGGGgrrr', 'rrrGrrrrrrGrrr', 'GGGGrrrrrrrrrr', 'rrrrrrrrrrGGGG',
                                       'rrrrGGgrrrrGGg', 'rrrrrrGrrrrrrG'],
                               '6.1': ['GGgrrGGGrrrGGGgrrrGGGg', 'rrGrrrrrrrrrrrGrrrrrrG', 'GGGrrrrrGGgrrrrGGgrrrr',
                                       'GGGrrrrrrrGrrrrrrGrrrr', 'rrrGGGrrrrrrrrrrrrGGGG', 'rrrGGGrrrrrGGGgrrrGGGg']}

    def get_neighbor_map(self):
        neighbor_dict = OrderedDict()
        for tls, data in self.tls_data_dict.items():
            neighbor_dict[tls] = self.tls_data_dict[tls][1]

        return neighbor_dict

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))

        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()

        return self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list, self.lane_links, list(self.num_links_lane)

    def save_net_config(self, config_data_save_path='./monaco_network_30/monaco_network_30_config.json', save_file=False):
        self.start_simulation(sumo_config_file_path='monaco_network_30/monaco_network_30.sumocfg')
        net_config = OrderedDict()
        self.tls_phase_dict = OrderedDict(self.tls_phase_dict)
        neighbor_dict = self.get_neighbor_map()
        for tls in list(self.tls_data_dict.keys()):
            in_lane_list, out_lane_list,  incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane = self.get_net_config(tls_id=tls)
            phase_codes = self.tls_phase_dict[self.tls_data_dict[tls][0]]
            valid_phase_list = phase_codes
            neighbor_list = neighbor_dict[tls]
            for i, in_detector in enumerate(incoming_detector_list):
                incoming_detector_list[i] = 'ild:' + in_detector
            for i, out_detector in enumerate(outgoing_detector_list):
                outgoing_detector_list[i] = 'ild:' + out_detector

            phase_type = self.tls_data_dict[tls][0]
            phase_type_index = list(self.tls_phase_dict.keys()).index(self.tls_data_dict[tls][0])
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]

            if tls == 'joinedS_0':
                tls_position_vec = [item for item in traci.junction.getPosition('cluster_9154_9919')]
            elif tls == 'joinedS_1':
                tls_position_vec = [item for item in traci.junction.getPosition('9316')]
            else:
                tls_position_vec = [item for item in traci.junction.getPosition(tls)]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)
            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_codes,
                'neighbor_list': neighbor_list,
                'phase_lane_state': None,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list
            }

        save_as_json(config_data_save_path, net_config)
        print("Save Monaco config data!")


class cologne_network_config_saver(net_config_saver):
    def __init__(self):
        super().__init__()

        self.tls_data_dict = {'252017285': ('2.0', ['252046468', '252016271', '252016274', '796761043']),
                              '32319828': ('2.1', ['252016281', '252016282']),
                              '62426694': ('3.0', ['256190118', '1679948681', '3008854750']),
                              '280120513': ('3.1', ['1679948681', '274333875', '256190139']),
                              '256201389': ('3.2', ['3302422976', '3588475451']),
                              '26110729': ('4.0', ['258347996', '247379907']),
                              '247379907': ('4.0', ['cluster_1098574052_1098574061_247379905', '26110729']),
                              'cluster_1098574052_1098574061_247379905': ('4.1', ['247379907', '258346770']),
                              }

        self.tls_phase_dict = {
            '2.0': ['rrrrGGggrrrrGGgg', 'GGggrrrrGGggrrrr'],
            '2.1': ['GGggGGgg', 'rrGGrrGG'],
            '3.0': ['GGgGggrrr', 'rrGrGGrrr', 'GrrrrrGGg'],
            '3.1': ['GggrrrGGg', 'rGGrrrrrG', 'rrrGGgGrr'],
            '3.2': ['rrrGGgGgg', 'rrrrrGrGG', 'GGgGrrrrr'],
            '4.0': ['rrrrGGGggrrrrGGGgg', 'rrrrrrrGGrrrrrrrGG', 'GGggrrrrrGGggrrrrr', 'rrGGrrrrrrrGGrrrrr'],
            '4.1': ['rrrrGGggrrrrGGgg', 'rrrrrrGGrrrrrrGG', 'GGggrrrrGGggrrrr', 'rrGGrrrrrrGGrrrr'],
        }

    def get_neighbor_map(self):
        neighbor_dict = OrderedDict()
        for tls, data in self.tls_data_dict.items():
            neighbor_dict[tls] = self.tls_data_dict[tls][1]

        return neighbor_dict

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))

        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()

        return self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list, self.lane_links, list(self.num_links_lane)

    def save_net_config(self, config_data_save_path='./cologne_network_8/cologne_network_8_config.json', save_file=False):
        self.start_simulation(sumo_config_file_path='cologne_network_8/cologne_network_8.sumocfg')

        net_config = OrderedDict()
        self.tls_phase_dict = OrderedDict(self.tls_phase_dict)
        neighbor_dict = self.get_neighbor_map()
        tls_list = list(self.tls_data_dict.keys())
        assert tls_list != traci.trafficlight.getIDList()
        for tls in list(self.tls_data_dict.keys()):
            in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane = self.get_net_config(tls_id=tls)
            phase_codes = self.tls_phase_dict[self.tls_data_dict[tls][0]]
            valid_phase_list = phase_codes
            neighbor_list = neighbor_dict[tls]
            phase_type = self.tls_data_dict[tls][0]
            phase_type_index = list(self.tls_phase_dict.keys()).index(self.tls_data_dict[tls][0])
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = [item for item in traci.junction.getPosition(tls)]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)
            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_codes,
                'neighbor_list': neighbor_list,
                'phase_lane_state': None,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list
            }

        if save_file:
            save_as_json(config_data_save_path, net_config)
            print("Save Cologne config data!")
        else:
            return 0


class ingolstadt_network_config_saver(net_config_saver):
    def __init__(self):
        super().__init__()
        self.extended_lane_dict = {
            # 1863241632
            '170018165#0_1': ['201238719#0_1'],
            '170018165#0_2': ['201238719#0_2'],
            '-170018165#1_1': ['-170018165#2.210_1'],
            '-170018165#1_2': ['-170018165#2.210_2'],
            '-170018165#1_3': ['-170018165#2.210_3'],

            # 2330725114
            '224251776#0_1': ['-224251774#0_1'],
            '224251776#0_2': ['-224251774#0_2'],

            # 243351999
            '233675413#3_1': ['233675413#2_1'],
            '233675413#3_2': ['233675413#2_1'],

            # 243641585 None
            # '-201201945#0.78_1': ['-201201945#0_1'],
            # '-201201945#0.78_2': ['-201201945#0_1'],

            # 243749571
            # '612075153#1_1': ['612075153#0_1', '-399835085#0_1', '-315358258_1'],
            '612075153#1_1': ['612075153#0_1'],

            # 30503246
            # '-4942389#0_1': ['-4942389#1_1', '-315358245_1'],
            # '-4942389#0_2': ['-4942389#1_2', '-315358245_1'],
            # '-4942389#0_3': ['-4942389#1_3', '-315358245_1'],
            '-4942389#0_1': ['-4942389#1_1'],
            '-4942389#0_2': ['-4942389#1_2'],
            '-4942389#0_3': ['-4942389#1_3'],

            # 30624898
            '315358251#1_1': ['315358242#3_1'],
            '315358251#1_2': ['315358242#3_1'],
            '315358251#1_3': ['315358242#3_2'],

            # 32564122
            '-24693977#0_1': ['-24693977#1_1'],
            '-24693977#0_2': ['-24693977#1_2'],
            '-24693977#0_3': ['-24693977#1_3'],

            # 89127267
            '-447569997#0_1': ['-447569998#1_1'],
            '-447569997#0_2': ['-447569998#1_2'],

            # 89173763
            '-201963533#0_1': ['-201963533#1.145_1'],
            '-201963533#0_2': ['-201963533#1.145_2'],
            '-201963533#0_3': ['-201963533#1.145_3'],
            '-18809672#1_1': ['-18809672#4_1'],

            # 89173808
            # '-25122731#1_1': ['-30482614_1', '-30482615#4_0'],
            '-25122731#1_1': ['-30482614_1'],

            # cluster_1427494838_273472399
            '315358250#1_1': ['315358250#0_1'],
            '315358250#1_2': ['315358250#0_2'],
            '315358250#1_3': ['315358250#0_3'],
            '-129379921#0_1': ['-129379921#1_1'],
            '-129379921#0_2': ['-129379921#1_2'],
            '-129379921#0_3': ['-129379921#1_3'],

            # cluster_1757124350_1757124352
            '124812856#1_1': ['124812856#0_1'],
            '124812856#1_2': ['124812856#0_1'],
            '124812856#1_3': ['124812856#0_2'],

            # cluster_1863241547_1863241548_1976170214
            '137133006#1_1': ['137133006#0_1'],
            '137133006#1_2': ['137133006#0_3'],
            '137133006#1_3': ['137133006#0_4'],

            '128361109#4_1': ['128361109#3_1'],
            '128361109#4_2': ['128361109#3_2'],
            '128361109#4_3': ['128361109#3_3'],
            '128361109#4_4': ['128361109#3_4'],

            '176550246_1': ['201201953#12_1'],
            '176550246_2': ['201201953#12_2'],
            '176550246_3': ['201201953#12_3'],
            '176550246_4': ['201201953#12_4'],

            # '201238726#0.117_1': ['201238726#0_1'],
            # '201238726#0.117_2': ['201238726#0_1'],
            # '201238726#0.117_3': ['201238726#0_1'],

            '201238729#1_1': ['201238729#3_1'],
            '201238729#1_2': ['201238729#3_2'],
            '201238730_1': ['176550249#0_1'],
            '201238730_2': ['176550249#0_2'],

            # cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190 None

            # gneJ143
            '10425609#1_1': ['10425609#0_1'],
            '10425609#1_2': ['10425609#0_2'],
            '10425609#1_3': ['10425609#0_3'],

            # gneJ207
            '164051413_1': ['653473569#5_1'],
            '164051413_2': ['653473569#5_2'],

            # gneJ208
            '286646456#0_1': ['224251774#1_1'],

            # gneJ210
            '51857517#1_1': ['51857517#0.33_1'],
            '51857517#1_2': ['51857517#0.33_2'],
            '51857517#1_3': ['51857517#0.33_3'],
            '51857517#1_4': ['51857517#0.33_4'],

            # gneJ255
            '168702040#4_1': ['168702040#3_1'],
            '168702040#4_2': ['168702040#3_2'],
            '168702040#4_3': ['168702040#3_3'],

            # gneJ257 None
        }

        self.replaced_lane_dict = {
            # 30503246
            # in
            '-4942389#0_1': ['-4942389#1_1'],
            '-4942389#0_2': ['-4942389#1_2'],
            '-4942389#0_3': ['-4942389#1_3'],
            # out
            '4942389#0_1': ['4942389#1_1'],
            '4942389#0_2': ['4942389#1_1'],

            # 30624898
            # in
            '315358251#1_1': ['315358242#3_1'],
            '315358251#1_2': ['315358242#3_1'],
            '315358251#1_3': ['315358242#3_2'],
            # out
            '-315358251#1_1': ['-315358251#0_1'],
            '-315358251#1_2': ['-315358251#0_2'],

            # 89127267
            # in
            '-447569997#0_1': ['-447569998#1_1'],
            '-447569997#0_2': ['-447569998#1_2'],

            # 89173763
            # in
            '-201963533#0_1': ['-201963533#0_1', '-201963533#1.145_1'],
            '-201963533#0_2': ['-201963533#0_2', '-201963533#1.145_1'],
            '-201963533#0_3': ['-201963533#0_3', '-201963533#1.145_1'],

            # cluster_1427494838_273472399
            # in
            '315358250#1_1': ['315358250#1_1', '315358250#0_1'],
            '315358250#1_2': ['315358250#1_2', '315358250#0_2'],
            '315358250#1_3': ['315358250#1_3', '315358250#0_3'],
            # out
            '-315358250#1_1': ['-315358250#1_1', '-315358250#0_1'],
            '-315358250#1_2': ['-315358250#1_2', '-315358250#0_2'],

            # cluster_1863241547_1863241548_1976170214
            # in
            '128361109#4_1': ['128361109#3_1'],
            '128361109#4_2': ['128361109#3_2'],
            '128361109#4_3': ['128361109#3_3'],
            '128361109#4_4': ['128361109#3_4'],
            '201238726#1_1': ['201238726#0.117_1'],
            '201238726#1_2': ['201238726#0.117_2'],
            '201238726#1_3': ['201238726#0.117_3'],
            '176550246_1': ['176550246_1', '201201953#12_1'],
            '176550246_2': ['176550246_2', '201201953#12_2'],
            '176550246_3': ['176550246_3', '201201953#12_3'],
            '176550246_4': ['176550246_4', '201201953#12_4'],

            # gneJ143
            '10425609#1_1': ['10425609#0_1'],
            '10425609#1_2': ['10425609#0_2'],
            '10425609#1_3': ['10425609#0_3']
        }

        self.extended_lane_length_dict = dict()
        self.replaced_lane_length_dict = dict()

    def get_neighbor_map(self):
        # The neighbor maps is not applicable for Ingolstadt network
        self.neighbor_map = {'1863241632': ['1863241639', '1863241614', '1782978746'],
                             '2330725114': ['2903266814', 'gneJ21'],
                             '243351999': ['2113472091', '274041328', '243636071'],
                             '243641585': ['497590188', 'cluster_1840209252_292578437', 'cluster_1840209209_268417350',
                                           'gneJ26', '243641589'],

                             '243749571': ['1509951406', 'gneJ25', '1137585395'],
                             '30503246': ['1840464316i', '1840464353', '2769732328', '30624898'],
                             '30624898': ['1840464393', '2769732343', '2769732333', '30503246'],
                             '32564122': ['247957651', '249176474', '32564123'],
                             '89127267': ['1517542714', '497590081', '3214665955'],
                             '89173763': ['2068408769', '89173755', '243636120', '356694630'],
                             '89173808': ['gneJ129', '3079046105', 'cluster_336359446_470954969', '364442178'],
                             'cluster_1427494838_273472399': ['30624890', '1427494830', '497590907'],
                             'cluster_1757124350_1757124352': ['1387938626', '370357925', 'gneJ136',
                                                               'cluster_1041665625_cluster_1387938793_1387938796_cluster_1757124361_1757124367_32564126'],
                             'cluster_1863241547_1863241548_1976170214': ['1782978693', '1863241614', '1602381826',
                                                                          '1976170215', '1602381830', '1602381828',
                                                                          'gneJ10', 'gneJ11'],
                             'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': [
                                 '1200363932', '1331204959', 'gneJ254', '1833941883', '1200363969', '1200363973'],
                             'gneJ143': [],
                             'gneJ207': [],
                             'gneJ208': [],
                             'gneJ210': [],
                             'gneJ255': [],
                             'gneJ257': [],
                             }
        raise NotImplementedError

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))

        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()

        # Get the traffic light program from traci
        traffic_phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].getPhases()
        self.phase_list = []
        for phase_tuple in traffic_phases:
            phase_code = phase_tuple.state
            if 'y' not in phase_code and 'G' in phase_code and phase_code not in self.phase_list:
                self.phase_list.append(phase_code)

        return self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list, self.lane_links, list(self.num_links_lane), self.phase_list

    def save_net_config(self, config_data_save_path='./ingolstadt_network_21/ingolstadt_network_21_config.json', save_file=False):
        self.start_simulation(sumo_config_file_path='ingolstadt_network_21/ingolstadt_network_21.sumocfg')

        net_config = OrderedDict()
        tls_list = traci.trafficlight.getIDList()
        for tls in tls_list:
            in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane, phase_list = self.get_net_config(tls_id=tls)

            neighbor_list = None

            # For the intersection 243641585 and cluster_1427494838_273472399, the phase codes are not aligned with the
            # length of the traffic movements, and needs a rectification
            if tls == '243641585':
                valid_phase_list = ['rrrrGgGGGG', 'rrrrGGrrrr', 'GGGGrrrrrr']
            elif tls == 'cluster_1427494838_273472399':
                # valid_phase_list = ['GGgrrGGG', 'GGGrrrrr', 'rrrrrrrr', 'rrrGGGrr']
                # Remove the all red phase
                phase_list = ['rrGGgrrGGG', 'rrGGGrrrrr', 'rrrrrGGGrr']
                valid_phase_list = ['GGgrrGGG', 'GGGrrrrr', 'rrrGGGrr']
            else:
                valid_phase_list = phase_list

            phase_type = str(len(phase_list))
            phase_type_index = tls_list.index(tls)
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = [None, None]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                if in_lane in self.extended_lane_dict:
                    for extended_lane in self.extended_lane_dict[in_lane]:
                        self.extended_lane_length_dict[extended_lane] = traci.lane.getLength(extended_lane)

                if in_lane in self.replaced_lane_dict:
                    if len(self.replaced_lane_dict[in_lane]) == 1:
                        self.replaced_lane_length_dict[in_lane] = traci.lane.getLength(self.replaced_lane_dict[in_lane][0])
                    else:
                        self.replaced_lane_length_dict[in_lane] = traci.lane.getLength(self.replaced_lane_dict[in_lane][0]) + traci.lane.getLength(self.replaced_lane_dict[in_lane][1])

                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)

            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                if out_lane in self.extended_lane_dict:
                    for extended_lane in self.extended_lane_dict[out_lane]:
                        self.extended_lane_length_dict[extended_lane] = traci.lane.getLength(extended_lane)
                if out_lane in self.replaced_lane_dict:
                    if len(self.replaced_lane_dict[out_lane]) == 1:
                        self.replaced_lane_length_dict[out_lane] = traci.lane.getLength(self.replaced_lane_dict[out_lane][0])
                    else:
                        self.replaced_lane_length_dict[out_lane] = traci.lane.getLength(self.replaced_lane_dict[out_lane][0]) + traci.lane.getLength(self.replaced_lane_dict[out_lane][1])

                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            extended_lane_dict = self.extended_lane_dict
            extended_lane_length_dict = self.extended_lane_length_dict

            replaced_lane_dict = self.replaced_lane_dict
            replaced_lane_length_dict = self.replaced_lane_length_dict

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_list,
                'neighbor_list': neighbor_list,
                'phase_lane_state': None,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list,
                'extended_lane_dict': extended_lane_dict,
                'extended_lane_length_dict': extended_lane_length_dict,
                'replaced_lane_dict': replaced_lane_dict,
                'replaced_lane_length_dict': replaced_lane_length_dict
            }

        if save_file:
            save_as_json(config_data_save_path, net_config)
            print("Save Ingolstadt config data!")
        else:
            pass


class shaoxing_network_config_saver(net_config_saver):
    def __init__(self):
        super().__init__()

    def get_neighbor_map(self):
        self.neighbor_map = {
            'J6': [None, None, 'J12', None],
            'J12': ['J6', None, 'J16', None],
            'J16': ['J12', None, 'J20', None],
            'J20': ['J16', None, 'J28', None],
            'J28': ['J20', None, 'J32', None],
            'J32': ['J28', None, 'J36', None],
            'J36': ['J32', None, None, None]
        }

        return self.neighbor_map

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        # Get incoming lane list and outgoing lane list
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))
        # Get lane links
        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        # Get phase list
        self.phase_list = []
        phase_program = traci.trafficlight.getAllProgramLogics(tlsID=tls_id)
        for phase in phase_program[0].phases:
            phase_code = phase.state
            if 'g' not in phase_code and 'G' not in phase_code:
                pass
            else:
                self.phase_list.append(phase_code)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()
        self.valid_phase_list = self.phase_list

        return self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list, self.lane_links, list(self.num_links_lane), self.phase_list, self.valid_phase_list

    def save_net_config(self, config_data_save_path='./shaoxing_network_7/shaoxing_network_7_config.json', save_file=False):
        self.start_simulation(sumo_config_file_path='shaoxing_network_7/shaoxing_network_7.sumocfg')

        net_config = OrderedDict()
        tls_list = ['J6', 'J12', 'J16', 'J20', 'J28', 'J32', 'J36']
        for tls in tls_list:
            in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane, phase_list, valid_phase_list = self.get_net_config(tls_id=tls)

            phase_lane_state = None
            neighbor_list = self.get_neighbor_map()[tls]

            phase_type = str(len(self.phase_list)) + '.'
            phase_type_index = tls_list.index(tls)
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = [None, None]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)

            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_list,
                'neighbor_list': neighbor_list,
                'phase_lane_state': phase_lane_state,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list
            }

        if save_file:
            save_as_json(config_data_save_path, net_config)
            print("Save Shaoxing config data!")
        else:
            pass


class shenzhen_network_config_saver(net_config_saver):
    def __init__(self):
        super().__init__()

    def get_neighbor_map(self):
        raise NotImplementedError

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        # Get incoming lane list and outgoing lane list
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))
        # Get lane links
        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        # Get phase list
        self.phase_list = []
        phase_program = traci.trafficlight.getAllProgramLogics(tlsID=tls_id)
        for phase in phase_program[0].phases:
            phase_code = phase.state
            if 'g' not in phase_code and 'G' not in phase_code:
                pass
            else:
                self.phase_list.append(phase_code)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()
        self.valid_phase_list = self.phase_list

        return self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list, self.lane_links, list(self.num_links_lane), self.phase_list, self.valid_phase_list

    def save_net_config(self, config_data_save_path='./shenzhen_network_55/shenzhen_network_55_config.json', save_file=False):
        self.start_simulation(sumo_config_file_path='shenzhen_network_55/shenzhen_network_55.sumocfg')

        net_config = OrderedDict()
        # All traffic lights in the network
        all_tls_list = list(traci.trafficlight.getIDList())
        for tls in all_tls_list:
            in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane, phase_list, valid_phase_list = self.get_net_config(tls_id=tls)

            phase_lane_state = None
            neighbor_list = None

            phase_type = str(len(self.phase_list)) + '.'
            phase_type_index = all_tls_list.index(tls)
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = [None, None]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)

            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_list,
                'neighbor_list': neighbor_list,
                'phase_lane_state': phase_lane_state,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list
            }

        if save_file:
            save_as_json(config_data_save_path, net_config)
            print("Save Shenzhen 55 config data!")
        else:
            pass

    def save_net_config_29(self, config_data_save_path='./shenzhen_network_29/shenzhen_network_29_config.json', save_file=False):
        self.start_simulation(sumo_config_file_path='shenzhen_network_29/shenzhen_network_29.sumocfg')

        net_config = OrderedDict()
        # All traffic lights in the network
        all_tls_list = list(traci.trafficlight.getIDList())
        # Considered traffic lights in the network
        tls_list = []
        for tls in all_tls_list:
            if 'GS' not in tls and tls != '1673431902':
                tls_list.append(tls)

        for tls in tls_list:
            in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list, lane_links, num_links_lane, phase_list, valid_phase_list = self.get_net_config(tls_id=tls)

            phase_lane_state = None
            neighbor_list = None

            phase_type = str(len(self.phase_list)) + '.'
            phase_type_index = all_tls_list.index(tls)
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = [None, None]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)

            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_list,
                'neighbor_list': neighbor_list,
                'phase_lane_state': phase_lane_state,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list
            }

        if save_file:
            save_as_json(config_data_save_path, net_config)
            print("Save Shenzhen 29 config data!")
        else:
            pass


class resco_network_config_saver(net_config_saver):
    def __init__(self, network_type):
        super().__init__()

        self.network_type = network_type

    def get_net_config(self, tls_id):
        """
        get the net configurations from the simulator
        :param tls_id: the id of the target traffic light
        :return: net configurations
        """
        # Get incoming lane list and outgoing lane list
        self.incoming_lane_list = []
        self.outgoing_lane_list = []
        self.lane_links = {}
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane in lanes:
            if lane not in self.incoming_lane_list:
                self.incoming_lane_list.append(lane)

        self.num_links_lane = np.zeros(len(self.incoming_lane_list))
        # Get lane links
        for i in range(len(self.incoming_lane_list)):
            self.num_links_lane[i] = traci.lane.getLinkNumber(self.incoming_lane_list[i])
            self.lane_links[self.incoming_lane_list[i]] = []
            outgoing_lanes_info = traci.lane.getLinks(self.incoming_lane_list[i])

            for info in outgoing_lanes_info:
                self.lane_links[self.incoming_lane_list[i]].append(info[0])

            for lane in self.lane_links[self.incoming_lane_list[i]]:
                if lane not in self.outgoing_lane_list:
                    self.outgoing_lane_list.append(lane)

        # Get phase list
        self.phase_list = []
        phase_program = traci.trafficlight.getAllProgramLogics(tlsID=tls_id)
        for index, phase in enumerate(phase_program[0].phases):
            # if 'g' not in phase_code and 'G' not in phase_code:
            #     pass
            # else:
            #     self.phase_list.append(phase_code)
            phase_code = phase.state
            if (index + 1) % 2 != 0:
                self.phase_list.append(phase_code)

        self.incoming_detector_list = self.incoming_lane_list.copy()
        self.outgoing_detector_list = self.outgoing_lane_list.copy()
        self.valid_phase_list = self.phase_list

        return (self.incoming_lane_list, self.outgoing_lane_list, self.incoming_detector_list, self.outgoing_detector_list,
                self.lane_links, list(self.num_links_lane), self.phase_list, self.valid_phase_list)

    def save_net_config(self, save_file=False):
        config_data_save_path = './{}/{}_config.json'.format(self.network_type, self.network_type)
        self.start_simulation(sumo_config_file_path='{}/{}.sumocfg'.format(self.network_type, self.network_type))

        net_config = OrderedDict()
        # All traffic lights in the network
        all_tls_list = list(traci.trafficlight.getIDList())
        for tls in all_tls_list:
            (in_lane_list, out_lane_list, incoming_detector_list, outgoing_detector_list,
             lane_links, num_links_lane, phase_list, valid_phase_list) = self.get_net_config(tls_id=tls)

            neighbor_list = None
            phase_lane_state = None

            phase_type = str(len(self.phase_list)) + '.'
            phase_type_index = all_tls_list.index(tls)
            binary_string = format(phase_type_index, '04b')
            phase_type_vec = [int(bit) for bit in binary_string]
            tls_position_vec = [None, None]

            lane_length_dict = OrderedDict()
            lane_max_speed_dict = OrderedDict()
            for in_lane in in_lane_list:
                lane_length_dict[in_lane] = traci.lane.getLength(in_lane)
                lane_max_speed_dict[in_lane] = traci.lane.getMaxSpeed(in_lane)

            for out_lane in out_lane_list:
                lane_length_dict[out_lane] = traci.lane.getLength(out_lane)
                lane_max_speed_dict[out_lane] = traci.lane.getMaxSpeed(out_lane)

            net_config[tls] = {
                'incoming_lane_list': in_lane_list,
                'outgoing_lane_list': out_lane_list,
                'incoming_detector_list': incoming_detector_list,
                'outgoing_detector_list': outgoing_detector_list,
                'lane_links': lane_links,
                'num_links_lane': num_links_lane,
                'action_space': phase_list,
                'neighbor_list': neighbor_list,
                'phase_lane_state': phase_lane_state,
                'phase_type': phase_type,
                'phase_type_vec': phase_type_vec,
                'tls_position_vec': tls_position_vec,
                'lane_length_dict': lane_length_dict,
                'lane_max_speed_dict': lane_max_speed_dict,
                'valid_phase_list': valid_phase_list
            }

        if save_file:
            save_as_json(config_data_save_path, net_config)
            print("Save grid4x4 config data!")
        else:
            pass


class resco_arterial_network_config_saver(net_config_saver):
    def __init__(self):
        super().__init__()

    def save_net_config(self, config_data_save_path):
        pass


if __name__ == '__main__':
    # grid_network_config = grid_network_config_saver()
    # grid_network_config.save_net_config(save_file=True)
    # monaco_net_config = monaco_network_config_saver()
    # monaco_net_config.save_net_config(save_file=True)
    # cologne_net_config = cologne_network_config_saver()
    # cologne_net_config.save_net_config(save_file=True)
    # ingolstadt_net_config = ingolstadt_network_config_saver()
    # ingolstadt_net_config.save_net_config(save_file=True)
    # shaoxing_net_config = shaoxing_network_config_saver()
    # shaoxing_net_config.save_net_config(save_file=True)
    # shenzhen_net_config = shenzhen_network_config_saver()
    # shenzhen_net_config.save_net_config(save_file=True)
    # shenzhen_net_config.save_net_config_29(save_file=True)
    # resco_network_config = resco_network_config_saver(network_type='grid_network_4_4')
    resco_network_config = resco_network_config_saver(network_type='arterial_network_4_4')
    resco_network_config.save_net_config(save_file=True)