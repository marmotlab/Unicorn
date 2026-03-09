from parameters import SUMO_PARAMS


def gen_cfg_file(path, seed=None, thread=None):
    map_name = path.split('/')[-1]
    if thread is None:
        sumocfg_file_path = path + '/{}.sumocfg'.format(map_name)
    else:
        sumocfg_file_path = path + ('/{}_{}.sumocfg'.format(map_name, thread))

    if map_name == 'grid_network_5_5':
        from maps.grid_network_5_5.build_file import gen_ma2c_grid_sumocfg_file
        gen_ma2c_grid_sumocfg_file(path=path,
                                   peak_flow1=SUMO_PARAMS.PEAK_FLOW1,
                                   peak_flow2=SUMO_PARAMS.PEAK_FLOW2,
                                   density=SUMO_PARAMS.INIT_DENSITY,
                                   seed=seed,
                                   thread=thread)

    elif map_name == 'monaco_network_30':
        from maps.monaco_network_30.build_file import gen_ma2c_monaco_sumocfg_file
        gen_ma2c_monaco_sumocfg_file(path=path, flow_rate=SUMO_PARAMS.FLOW, thread=thread)

    elif map_name == 'cologne_network_8':
        from maps.cologne_network_8.build_file import gen_cologne_sumocfg_file
        gen_cologne_sumocfg_file(path=path, seed=seed, thread=thread)

    elif map_name == 'ingolstadt_network_21':
        from maps.ingolstadt_network_21.build_file import gen_ingolstadt_sumocfg_file
        gen_ingolstadt_sumocfg_file(path=path, seed=seed, thread=thread)

    elif map_name == 'shaoxing_network_7':
        from maps.shaoxing_network_7.build_file import gen_shaoxing_sumocfg_file
        gen_shaoxing_sumocfg_file(path=path, seed=seed, thread=thread)

    elif map_name == 'shenzhen_network_29':
        from maps.shenzhen_network_29.build_file import gen_shenzhen_sumocfg_file
        gen_shenzhen_sumocfg_file(path=path, seed=seed, thread=thread)

    elif map_name == 'shenzhen_network_55':
        from maps.shenzhen_network_55.build_file import gen_shenzhen_sumocfg_file
        gen_shenzhen_sumocfg_file(path=path, seed=seed, thread=thread)

    elif map_name == 'arterial_network_4_4':
        from maps.arterial_network_4_4.build_file import gen_arterial_network_4_4_sumocfg_file
        gen_arterial_network_4_4_sumocfg_file(path=path, seed=seed, thread=thread)

    elif map_name == 'grid_network_4_4':
        from maps.grid_network_4_4.build_file import gen_grid_network_4_4_sumocfg_file
        gen_grid_network_4_4_sumocfg_file(path=path, seed=seed, thread=thread)

    else:
        raise NotImplementedError

    return sumocfg_file_path


if __name__ == '__main__':
    sumocfg_file = gen_cfg_file(path='grid_network_5_5', seed=1, thread=0)
