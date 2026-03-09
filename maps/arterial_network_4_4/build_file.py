import json
import traci

from sumolib import checkBinary


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def gen_arterial_network_4_4_sumocfg_file(path, seed, thread):
    if thread is None:
        sumocfg_file = 'arterial_network_4_4.sumocfg'
    else:
        sumocfg_file = '/arterial_network_4_4_{}.sumocfg'.format(int(thread))

    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="arterial_network_4_4.net.xml"/>\n'
    str_config += '	   <route-files value="arterial_network_4_4.rou.xml"/>\n'
    str_config += '    <additional-files value="arterial_network_4_4.add.xml"/>'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'

    write_file(path + sumocfg_file, str_config)

    print('Generate Arterial4x4 sumocfg file: {}.'.format(sumocfg_file))


def gen_arterial_network_4_4_add_file():
    sumo_add_save_path = 'arterial_network_4_4.add.xml'
    sumo_config_path = 'arterial_network_4_4_config.json'
    sumo_config_file_path = 'arterial_network_4_4.sumocfg'

    sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumo_config_file_path,
                 "--start",
                 "--quit-on-end"])

    with open(sumo_config_path, 'r+') as file:
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
        string += '  <laneAreaDetector file="ild.out" freq="1" id="{}" lane="{}" pos="-50" endPos="-1"/>\n'.format(lane,
                                                                                                                   lane)
    string += '</additional>'

    with open(sumo_add_save_path, 'w') as f:
        f.write(string)


if __name__ == '__main__':
    # gen_arterial_network_4_4_sumocfg_file(path='./', seed=0, thread=0)
    gen_arterial_network_4_4_add_file()
