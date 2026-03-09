from utils import save_as_csv
from collections import OrderedDict
from parameters import SUMO_PARAMS


class Evaluator:
    def __init__(self, env, exp_dir, agent_name):
        super(Evaluator, self).__init__()
        self.env = env
        self.exp_dir = exp_dir
        self.agent_name = agent_name

        self.data_file = 'eval_data'
        self.data_path = self.exp_dir + '/' + self.data_file

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.episode_step = None
        self.episode_reward = None
        self.episode_action_change = None

    def reset(self):
        # Reset evaluator params
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_action_change = 0

        # Reset Environment params
        self.env.reset_vars()

        return self.env.observe()

    def evaluate(self, index, seed):
        """
        Evaluate the trained model
        @param index: int -> test/evaluation index
        @param seed: seed
        @return:
        """
        # Reset simulations
        self.env.curr_episode = index
        self.env.set_world(seed=seed)
        # Force each agent to choose the first phase at the initial step
        for tls in self.env.tls_list:
            self.env.tls_dict[tls].set_green_phase(0, self.env.sumo_steps_green_phase)
        obs_n = self.reset()
        done = False
        while not done:
            if self.agent_name == 'FIXED':
                action_dict = OrderedDict()
                for i, tls in enumerate(self.env.tls_list):
                    action_dict[tls] = self.env.tls_dict[tls].get_fixed_time_action()
            elif self.agent_name == 'GREEDY':
                action_dict = OrderedDict()
                if self.env.tls_map_dataset == 'resco':
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_greedy_action(detector=False)
                else:
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_greedy_action(detector=True)

            elif self.agent_name == 'PRESSURE':
                action_dict = OrderedDict()
                if self.env.tls_map_dataset == 'resco':
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_pressure_action(detector=False)
                else:
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_pressure_action(detector=True)

            else:
                raise NotImplementedError

            next_obs_n, r_n, done, info = self.env.step(action_dict)

            obs_n = next_obs_n
            self.episode_step += 1
            self.episode_reward += info[0]
            self.episode_action_change += info[1]

        # collect trip information each run
        self.env.collect_trip_data()

        print('{} || Episode Length: {} || Episode Reward: {} || Action Change: {}'.format(index,
                                                                                           self.episode_step,
                                                                                           self.episode_reward,
                                                                                           self.episode_action_change / self.episode_step))

    def output_eval_data(self):
        """
        Save the evaluation results (traffic data and trip data)
        @return:
        """
        map_name_part = SUMO_PARAMS.NET_NAME.split('_')
        map_name = map_name_part[0] + '_' + map_name_part[1]
        traffic_data_path = self.data_path + '/' + '{}_{}_traffic.csv'.format(map_name, self.agent_name)
        trip_data_path = self.data_path + '/' + '{}_{}_trip.csv'.format(map_name, self.agent_name)
        save_as_csv(file=traffic_data_path, data=self.env.traffic_data)
        save_as_csv(file=trip_data_path, data=self.env.trip_data)


if __name__ == '__main__':
    import os
    import torch
    import random
    import numpy as np

    from utils import set_env

    random_seed = 21
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    seed = 100
    test_num = 10
    start_seed = 1000
    exp_dir = './Test'

    # Testing parameter for learning based method
    agent_name_list = ['FIXED', 'GREEDY', 'PRESSURE']  # ['FIXED', 'GREEDY', 'PRESSURE']
    # Learning Test
    for agent_name in agent_name_list:
        env = set_env(server_number=66, test=True)
        evaluator = Evaluator(env=env,
                              exp_dir=exp_dir,
                              agent_name=agent_name)

        seeds = [start_seed + seed * i for i in range(test_num)]
        print("Random Seed:{}".format(seeds))
        for i, seed in enumerate(seeds):
            evaluator.evaluate(index=i, seed=seed)
        evaluator.output_eval_data()
        print('Finish output the evaluation data for {}!'.format(agent_name))
