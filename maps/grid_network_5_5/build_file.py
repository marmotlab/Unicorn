import json
import numpy as np

from maps import randomTrips
from parameters import SUMO_PARAMS

MAX_CAR_NUM = 30


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def gen_random_rou_file(net_path, route_path, seed):
    randomTrips.main(randomTrips.get_options([
        '--net-file', net_path,
        '--output-trip-file', route_path,
        '--seed', str(seed),  # make runs reproducible
        '--end', str(SUMO_PARAMS.END_TIME),
        '--period', str(SUMO_PARAMS.PERIOD),
        '--min-distance', str(SUMO_PARAMS.MIN_DIS),  # prevent trips that start and end on the same edge
        '--fringe-factor', str(SUMO_PARAMS.FRINGE_FACTOR),
        '--trip-attributes', "departLane=\"best\" departSpeed=\"random\" departPos=\"random\""]))


def init_routes(density):
    init_flow = '  <flow id="i_%s" departPos="random_free" from="%s" to="%s" begin="0" end="1" departLane="%d" departSpeed="0" number="%d" type="type1"/>\n'
    output = ''
    in_nodes = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1,
                1, 2, 3, 4, 5, 25, 24, 23, 22, 21]
    out_nodes = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20,
                 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    # external edges
    sink_edges = []
    for i, j in zip(in_nodes, out_nodes):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        sink_edges.append('%s_%s' % (node1, node2))

    def get_od(node1, node2, k, lane=0):
        source_edge = '%s_%s' % (node1, node2)
        sink_edge = np.random.choice(sink_edges)
        return init_flow % (str(k), source_edge, sink_edge, lane, car_num)

    # streets
    k = 1
    car_num = int(MAX_CAR_NUM * density)
    for i in range(1, 25, 5):
        for j in range(4):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 1)
            output += get_od(node1, node2, k)
            k += 1
            output += get_od(node2, node1, k)
            k += 1
            output += get_od(node1, node2, k, lane=1)
            k += 1
            output += get_od(node2, node1, k, lane=1)
            k += 1
    # avenues
    for i in range(1, 6):
        for j in range(0, 20, 5):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 5)
            output += get_od(node1, node2, k)
            k += 1
            output += get_od(node2, node1, k)
            k += 1
    return output


def get_external_od(out_edges, dest=True):
    edge_maps = [0, 1, 2, 3, 4, 5, 5, 10, 15, 20, 25,
                 25, 24, 23, 22, 21, 21, 16, 11, 6, 1]
    cur_dest = []
    for out_edge in out_edges:
        in_edge = edge_maps[out_edge]
        in_node = 'nt' + str(in_edge)
        out_node = 'np' + str(out_edge)
        if dest:
            edge = '%s_%s' % (in_node, out_node)
        else:
            edge = '%s_%s' % (out_node, in_node)
        cur_dest.append(edge)
    return cur_dest


def output_flows(peak_flow1, peak_flow2, density, seed=None):
    """
    flow1: x11, x12, x13, x14, x15 -> x1, x2, x3, x4, x5
    flow2: x16, x17, x18, x19, x20 -> x6, x7, x8, x9, x10
    flow3: x1, x2, x3, x4, x5 -> x15, x14, x13, x12, x11
    flow4: x6, x7, x8, x9, x10 -> x20, x19, x18, x17, x16
    """
    if seed is not None:
        np.random.seed(seed)
    ext_flow = '  <flow id="f_%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    str_flows = '<routes>\n'
    str_flows += '  <vType id="type1" length="5" accel="5" decel="10"/>\n'
    # initial traffic dist
    if density > 0:
        str_flows += init_routes(density)

    # create external origins and destinations for flows
    srcs = []
    srcs.append(get_external_od([12, 13, 14], dest=False))
    srcs.append(get_external_od([16, 18, 20], dest=False))
    srcs.append(get_external_od([2, 3, 4], dest=False))
    srcs.append(get_external_od([6, 8, 10], dest=False))

    sinks = []
    sinks.append(get_external_od([2, 3, 4]))
    sinks.append(get_external_od([6, 8, 10]))
    sinks.append(get_external_od([14, 13, 12]))
    sinks.append(get_external_od([20, 18, 16]))

    # create volumes per 5 min for flows
    ratios1 = np.array([0.4, 0.7, 0.9, 1.0, 0.75, 0.5, 0.25])  # start from 0
    ratios2 = np.array([0.3, 0.8, 0.9, 1.0, 0.8, 0.6, 0.2])  # start from 15min
    flows1 = peak_flow1 * 0.6 * ratios1
    flows2 = peak_flow1 * ratios1
    flows3 = peak_flow2 * 0.6 * ratios2
    flows4 = peak_flow2 * ratios2
    flows = [flows1, flows2, flows3, flows4]
    times = np.arange(0, 3001, 300)
    id1 = len(flows1)
    id2 = len(times) - 1 - id1
    for i in range(len(times) - 1):
        name = str(i)
        t_begin, t_end = times[i], times[i + 1]
        # external flow
        k = 0
        if i < id1:
            for j in [0, 1]:
                for e1, e2 in zip(srcs[j], sinks[j]):
                    cur_name = name + '_' + str(k)
                    str_flows += ext_flow % (cur_name, e1, e2, t_begin, t_end, flows[j][i])
                    k += 1
        if i >= id2:
            for j in [2, 3]:
                for e1, e2 in zip(srcs[j], sinks[j]):
                    cur_name = name + '_' + str(k)
                    str_flows += ext_flow % (cur_name, e1, e2, t_begin, t_end, flows[j][i - id2])
                    k += 1
    str_flows += '</routes>\n'
    return str_flows


def gen_ma2c_rou_file(route_path, peak_flow1, peak_flow2, density, seed):
    write_file(route_path, output_flows(peak_flow1, peak_flow2, density, seed=seed))


def output_config(thread):
    net_file_name = 'grid_network_5_5.net.xml'
    if thread is None:
        route_file_name = 'grid_network_5_5.rou.xml'
    else:
        route_file_name = 'grid_network_5_5_{}.rou.xml'.format(int(thread))
    add_file_name = 'grid_network_5_5.add.xml'

    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="{}"/>\n'.format(net_file_name)
    str_config += '    <route-files value="{}"/>\n'.format(route_file_name)
    str_config += '    <additional-files value="{}"/>\n'.format(add_file_name)
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'

    return str_config


def gen_ma2c_grid_sumocfg_file(path, peak_flow1, peak_flow2, density, seed=None, thread=None):
    if thread is None:
        flow_file = 'grid_network_5_5.rou.xml'
        sumocfg_file = 'grid_network_5_5.sumocfg'
    else:
        flow_file = '/grid_network_5_5_{}.rou.xml'.format(int(thread))
        sumocfg_file = '/grid_network_5_5_{}.sumocfg'.format(thread)

    gen_ma2c_rou_file(path + flow_file, peak_flow1, peak_flow2, density, seed)

    sumocfg_file = path + sumocfg_file
    write_file(sumocfg_file, output_config(thread=thread))

    print('Generate Grid 5x5 sumocfg file: {}.'.format(sumocfg_file))

    return sumocfg_file


def gen_ma2c_grid_add_file(config_path, save_path):
    with open(config_path, 'r+') as file:
        content = file.read()

    config_data = json.loads(content)
    lane_list = set()
    for tls, data in config_data.items():
        for in_lane in data['incoming_lane_list']:
            lane_list.add(in_lane)
        for out_lane in data['outgoing_lane_list']:
            lane_list.add(out_lane)
    string = '<additional>\n'
    for i, lane in enumerate(list(lane_list)):
        string += '  <laneAreaDetector file="ild.out" freq="1" id="{}" lane="{}" pos="-50" endPos="-1"/>\n'.format(lane, lane)
    string += '</additional>'

    with open(save_path, 'w') as f:
        f.write(string)


if __name__ == '__main__':
    # gen_ma2c_grid_add_file(config_path='./grid_network_5_5_config.json', save_path='./grid_network_5_5.add1.xml')
    gen_ma2c_grid_sumocfg_file('/', 1100, 925, 0, None, 0)