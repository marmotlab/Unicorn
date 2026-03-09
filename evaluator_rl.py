from collections import OrderedDict
from parameters import SUMO_PARAMS
from torch.distributions.categorical import Categorical
from utils import convert_to_tensor, save_as_csv


class Evaluator:
    def __init__(self, env, model, exp_dir, model_name):
        super(Evaluator, self).__init__()
        self.env = env
        self.exp_dir = exp_dir
        self.local_network = model
        self.model_name = model_name

        self.data_file = 'eval_data'
        self.data_path = self.exp_dir + '/' + self.data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_network.to(self.device)

        self.episode_step = None
        self.episode_reward = None
        self.episode_action_change = None

        self.rnn_states_actor = None
        self.rnn_states_critic = None

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def load_model(self, model_path):
        """
        Load saved trained model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.local_network.load_state_dict(checkpoint['model_state_dict'])

    def reset(self):
        # Reset evaluator params
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_action_change = 0

        self.rnn_states_actor = None
        self.rnn_states_critic = None

        # Reset Environment params
        self.env.reset_vars()

        # Force each agent to choose the first phase at the initial step
        for tls in self.env.tls_list:
            self.env.tls_dict[tls].set_green_phase(0, self.env.sumo_steps_green_phase)

        obs_n = self.env.observe()

        return obs_n

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
        obs_n = self.reset()
        done = False
        while not done:
            if self.model_name == 'UNICORN':
                action_dict = OrderedDict()
                phase_vec_dict, phase_mask_dict = self.env.get_phase_vec_mask_dict()
                int_attr_vec_dict = self.env.get_int_attr_vec_dict()

                multi_agent_obs = convert_to_tensor(data=np.array(list(obs_n.values())), data_type=torch.float32,
                                                    device=self.device)
                multi_agent_phase_vec = convert_to_tensor(data=np.array(list(phase_vec_dict.values())),
                                                          data_type=torch.float32, device=self.device)
                multi_agent_phase_mask = convert_to_tensor(data=np.array(list(phase_mask_dict.values())),
                                                           data_type=torch.float32, device=self.device)
                multi_agent_int_attr_vec = convert_to_tensor(data=np.array(list(int_attr_vec_dict.values())),
                                                             data_type=torch.float32, device=self.device)
                with torch.no_grad():
                    multi_agent_policy, self.rnn_states_actor, _, _, _ = self.local_network(multi_agent_obs,
                                                                                            multi_agent_phase_vec,
                                                                                            multi_agent_int_attr_vec,
                                                                                            multi_agent_phase_mask,
                                                                                            self.rnn_states_actor)
                    multi_agent_policy_dist = Categorical(multi_agent_policy)
                    multi_agent_action = multi_agent_policy_dist.sample().reshape(-1)
                    multi_agent_logprob_p = multi_agent_policy_dist.log_prob(multi_agent_action).reshape(-1)
                action_list = multi_agent_action.detach().cpu().clone().numpy().flatten()
                for i, tls in enumerate(self.env.rl_tls_list):
                    if action_list[i] >= len(self.env.tls_dict[tls].action_space):
                        policy_dist = Categorical(multi_agent_policy[:, i, :].unsqueeze(0))
                        while action_list[i] >= len(self.env.tls_dict[tls].action_space):
                            action = policy_dist.sample().reshape(-1)
                            logprob_p = policy_dist.log_prob(action).reshape(-1)
                            action_list[i] = action.detach().cpu().clone().numpy().flatten()
                            multi_agent_action[i] = action
                            multi_agent_logprob_p[i] = logprob_p
                    action_dict[tls] = action_list[i]

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
        traffic_data_path = self.data_path + '/' + '{}_{}_traffic.csv'.format(map_name, self.model_name)
        trip_data_path = self.data_path + '/' + '{}_{}_trip.csv'.format(map_name, self.model_name)
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
    exp_dir = './Test'

    # Testing parameter for learning based method
    agent_name_list = ['UNICORN']  # ['UNICORN']
    model_path_list = [None] # To be set by the user
    # Learning Test
    for model_name, model_path in zip(agent_name_list, model_path_list):
        env = set_env(server_number=66, test=True)
        if model_name == 'UNICORN':
            from models.Unicorn import Unicorn
            model = Unicorn(input_dim=env.tls_obs_space,
                            agent_dim=env.tls_agent_space,
                            int_vec_dim=env.tls_int_attr_space,
                            actor_lr=1e-4,
                            critic_lr=1e-4)

        else:
            raise NotImplementedError

        evaluator = Evaluator(env=env,
                              model=model,
                              exp_dir=exp_dir,
                              model_name=model_name)

        evaluator.load_model(model_path)
        seeds = [1000 + seed * i for i in range(test_num)]
        print("Random Seed:{}".format(seeds))
        for i, seed in enumerate(seeds):
            evaluator.evaluate(index=i, seed=seed)

        evaluator.output_eval_data()
        print('Finish output the evaluation data!')
