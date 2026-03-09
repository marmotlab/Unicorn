
def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def gen_shaoxing_sumocfg_file(path, seed, thread):
    if thread is None:
        sumocfg_file = 'shaoxing_network_7.sumocfg'
    else:
        sumocfg_file = '/shaoxing_network_7_{}.sumocfg'.format(int(thread))

    str_content = '<?xml version="1.0" encoding="UTF-8"?>\n\n'
    str_content += '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n'
    str_content += '    <input>\n'
    str_content += '        <net-file value="fenglin_y2z_t.net.xml"/>\n'
    str_content += '        <route-files value="fenglin_y2z_t.rou.xml"/>\n'
    str_content += '        <additional-files value="E1_add.xml, e2detectors.add.xml, E3_add.xml"/>\n\n'
    str_content += '    </input>\n\n'
    str_content += '    <output>\n'
    str_content += '        <write-license value="true"/>\n'
    str_content += '        <fcd-output value="fcd.xml"/>\n'
    str_content += '        <fcd-output.signals value="true"/>\n'
    str_content += '    </output>\n\n'
    str_content += '    <processing>\n'
    str_content += '        <default.speeddev value="0"/>\n'
    str_content += '    </processing>\n\n'
    str_content += '    <report>\n'
    str_content += '        <duration-log.disable value="true"/>\n'
    str_content += '        <no-step-log value="true"/>\n'
    str_content += '    </report>\n\n'
    str_content += '</configuration>'

    write_file(path+sumocfg_file, str_content)

    print('Generate Shaoxing sumocfg file: {}.'.format(path+sumocfg_file))


if __name__ == '__main__':
    gen_shaoxing_sumocfg_file(path='/', seed=0, thread=0)


