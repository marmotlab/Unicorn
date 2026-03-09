

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def gen_ingolstadt_sumocfg_file(path, seed, thread):
    if thread is None:
        sumocfg_file = 'ingolstadt_network_21.sumocfg'
    else:
        sumocfg_file = '/ingolstadt_network_21_{}.sumocfg'.format(int(thread))

    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="ingolstadt_network_21.net.xml"/>\n'
    str_config += '	   <route-files value="ingolstadt_network_21.rou.xml"/>\n'
    str_config += '    <additional-files value="ingolstadt_network_e1.add.xml, ingolstadt_network_e2.add.xml"/>'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="57600"/>\n    <end value="61200"/>\n'
    str_config += '  </time>\n</configuration>\n'

    write_file(path+sumocfg_file, str_config)

    print('Generate Ingolstadt sumocfg file: {}.'.format(sumocfg_file))


if __name__ == '__main__':
    gen_ingolstadt_sumocfg_file(path='/', seed=0, thread=0)
