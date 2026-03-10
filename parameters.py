import datetime

date = datetime.datetime.now()


class INPUT_PARAMS:
    MAX_EPISODES        = 3000
    NUM_META_AGENTS     = 6
    OBS_TYPE            = 'V1'  # ['V1', 'V2']
    UPDATE_TYPE         = 'PPO_MAC'
    LOAD_MODEL          = False
    EXPERIMENT_PATH     = None
    CO_TRAIN            = False


class EXPERIMENT_PARAMS:
    BASE_PATH           = 'Train_MATSC'
    EXPERIMENT_NAME     = '{}_{}_{}_{}_UNICORN'.format(date.year, date.month, date.day, date.hour)
    EXPERIMENT_PATH     = './{}/{}'.format(BASE_PATH, EXPERIMENT_NAME)
    MODEL_PATH          = EXPERIMENT_PATH + '/model'
    GIFS_PATH           = EXPERIMENT_PATH + '/gifs'
    TRAIN_PATH          = EXPERIMENT_PATH + '/train'
    TRIP_PATH           = EXPERIMENT_PATH + '/trip_info'
    CONFIG_FILE_PATH    = EXPERIMENT_PATH + '/config.json'


class SUMO_PARAMS:
    # MA2C DATASETS: 'grid_network_5_5', 'monaco_network_30',
    # RESCO DATASETS: 'cologne_network_8', 'ingolstadt_network_21', 'arterial_network_4_4', 'grid_network_4_4'
    # GESA DATASETS: 'shaoxing_network_7', 'shenzhen_network_29', 'shenzhen_network_55'
    ALL_DATASETS        = ['cologne_network_8', 'ingolstadt_network_21', 'arterial_network_4_4',
                           'grid_network_4_4', 'shaoxing_network_7', 'shenzhen_network_29']
    NET_NAME            = 'grid_network_5_5' # ['grid_network_5_5', 'monaco_network_30']
    NET_PATH            = './maps/{}'.format(NET_NAME)
    CONFIG_PATH         = NET_PATH + '/{}_config.json'.format(NET_NAME)
    CO_TRAIN            = INPUT_PARAMS.CO_TRAIN
    GUI                 = False
    RANDOM_SEED         = True
    OBS_SHARING         = False
    REWARD_DETECTOR     = True
    REGIONAL_REWARD     = True
    MAX_DISTANCE        = None
    SEED                = 1
    MAX_SUMO_STEP       = 3600
    MAX_TEST_STEP       = 3600
    # MA2C: 5s/2s, RESCO and SG: 10s/3s (5s/2s for arterial4x4), GESA: 10s/5s
    GREEN_DURATION      = 15
    YELLOW_DURATION     = 5
    TELEPORT_TIME       = 300  # (300 for ma2c and sg dataset, 600 for GESA dataset, -1 for RESCO dataset)
    # random route
    END_TIME            = 300
    PERIOD              = 2
    MIN_DIS             = 200
    FRINGE_FACTOR       = 3
    # baseline route for grid 5x5 grid network (default: 1100, 925)
    PEAK_FLOW1          = 1100
    PEAK_FLOW2          = 925
    INIT_DENSITY        = 0
    # baseline route for 30 monaco network (default: 325)
    FLOW                = 325
    # baseline route for 16 singapore network
    SG_FLOW             = 1100


class TRAIN_PARAMS:
    MAX_EPISODE         = INPUT_PARAMS.MAX_EPISODES
    NUM_META_AGENTS     = INPUT_PARAMS.NUM_META_AGENTS
    LOAD_MODEL          = INPUT_PARAMS.LOAD_MODEL
    EXPERIMENT_PATH     = INPUT_PARAMS.EXPERIMENT_PATH
    CONTRASTIVE_LOSS    = False
    NEIGHBOR_REWARD     = False
    WANDB               = False
    USE_GPU             = True
    NUM_GPU             = 1
    SAVE_MODEL_STEP     = 500
    SUMMARY_WINDOW      = 30
    RESET_OPTIM         = False
    RESET_OPTIM_STEP    = 20000


class NETWORK_PARAMS:
    CO_TRAIN            = INPUT_PARAMS.CO_TRAIN
    OBS_TYPE            = INPUT_PARAMS.OBS_TYPE
    UPDATE_TYPE         = INPUT_PARAMS.UPDATE_TYPE
    VL_FACTOR           = 0.5
    EL_FACTOR           = 2e-3
    PL_FACTOR           = 2e-4
    CL_FACTOR           = 1e-5
    CONTRASTIVE_TEMP    = 0.2
    CONTRASTIVE_BATCH   = 256
    GAMMA               = .95
    LAMBDA_ADV          = .98
    A_LR_Q              = 1.e-4
    C_LR_Q              = 2.e-4
    GRAD_CLIP           = 10
    EPS_CLIP            = 0.2
    K_EPOCH             = 6


class JOB_OPTIONS:
    GET_EXPERIENCE       = 1
    GET_GRADIENT         = 2


class COMPUTE_OPTIONS:
    MULTI_THREADED       = 1
    SINGLE_THREADED      = 2


class ALGORITHM_OPTIONS:
    A3C                  = 1
    PPO                  = 2


JOB_TYPE                 = JOB_OPTIONS.GET_EXPERIENCE
COMPUTE_TYPE             = COMPUTE_OPTIONS.SINGLE_THREADED
ALGORITHM_TYPE           = ALGORITHM_OPTIONS.PPO