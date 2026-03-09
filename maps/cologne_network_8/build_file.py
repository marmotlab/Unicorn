

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def gen_cologne_sumocfg_file(path, seed, thread):
    if thread is None:
        sumocfg_file = 'cologne_network_8.sumocfg'
    else:
        sumocfg_file = '/cologne_network_8_{}.sumocfg'.format(int(thread))

    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="cologne_network_8.net.xml"/>\n'
    str_config += '	   <route-files value="cologne_network_8.rou.xml"/>\n'
    str_config += '    <additional-files value="cologne_network_e1.add.xml, cologne_network_e2.add.xml"/>'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="25200"/>\n    <end value="28800"/>\n'
    str_config += '  </time>\n</configuration>\n'

    write_file(path+sumocfg_file, str_config)

    print('Generate Cologne sumocfg file: {}.'.format(sumocfg_file))


if __name__ == '__main__':
    gen_cologne_sumocfg_file(path='/', seed=0, thread=0)
