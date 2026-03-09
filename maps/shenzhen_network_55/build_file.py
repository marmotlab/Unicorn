

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def gen_shenzhen_sumocfg_file(path, seed, thread):
    if thread is None:
        sumocfg_file = 'shenzhen_network_55.sumocfg'
    else:
        sumocfg_file = '/shenzhen_network_55_{}.sumocfg'.format(thread)

    str_content = '<?xml version="1.0" encoding="UTF-8"?>\n\n'
    str_content += '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n\n'
    str_content += '    <input>\n'
    str_content += '        <net-file value="osm.net.xml"/>\n'
    str_content += '        <route-files value="osm_pt.rou.xml,osm.passenger.trips.xml,osm.truck.trips.xml,osm.bus.trips.xml"/>\n'
    str_content += '        <additional-files value="osm.poly.xml,osm_stops.add.xml,e1detectors.add.xml,e2detectors.add.xml"/>\n'
    str_content += '    </input>\n\n'
    str_content += '    <processing>\n'
    str_content += '        <ignore-route-errors value="true"/>\n'
    str_content += '    </processing>\n\n'
    str_content += '    <routing>\n'
    str_content += '        <device.rerouting.adaptation-steps value="18"/>\n'
    str_content += '        <device.rerouting.adaptation-interval value="10"/>\n'
    str_content += '    </routing>\n\n'
    str_content += '    <report>\n'
    str_content += '        <verbose value="true"/>\n'
    str_content += '        <duration-log.statistics value="true"/>\n'
    str_content += '        <no-step-log value="true"/>\n'
    str_content += '    </report>\n\n'
    str_content += '    <gui_only>\n'
    str_content += '        <gui-settings-file value="osm.view.xml"/>\n'
    str_content += '    </gui_only>\n\n'
    str_content += '</configuration>'

    write_file(path+sumocfg_file, str_content)

    print('Generate Shenzhen sumocfg: {}.'.format(sumocfg_file))


if __name__ == '__main__':
    gen_shenzhen_sumocfg_file(path='/', seed=0, thread=0)


