import ray
import time
import torch
import numpy as np

from parameters import *
from collections import OrderedDict
from models.HeteroLight import HeteroLight
from torch.distributions.categorical import Categorical
from utils import set_env, convert_to_tensor, calculate_discount_return


@ray.remote(num_cpus=1, num_gpus=0)
class Runner(object):
    """ Actor object to start running simulation on workers.
        Gradient computation is also executed on this object."""

    def __init__(self, meta_agent_id):
        self.id = meta_agent_id
        self.device = torch.device('cpu')
        self.curr_episode = int(meta_agent_id)

        self.env = set_env(server_number=meta_agent_id)

        self.input_size = self.env.tls_obs_space
        self.output_size = self.env.tls_action_space
        self.local_network = HeteroLight(input_dim=self.env.tls_obs_space,
                                         agent_dim=self.env.tls_agent_space,
                                         int_vec_dim=self.env.tls_int_attr_space,
                                         actor_lr=1e-4, critic_lr=1e-4).to(self.device)

        self.experience_buffers = None
        self.return_buffers = None
        self.bootstrap_values = None
        self.rnn_states_actor = None
        self.rnn_states_critic = None

        self.episode_step = None
        self.episode_reward = None
        self.episode_action_change = None
        self.episode_eval_metrics = None

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def initialization(self):
        # Initialize variables
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_action_change = 0

        # Initialize dicts
        self.experience_buffers = [[] for _ in range(7)]
        self.return_buffers = [[] for _ in range(9)]
        self.bootstrap_values = None
        self.rnn_states_actor = None
        self.rnn_states_critic = None

    def run_episode_single_threaded(self):
        # Initialize variables and buffer
        start_time = time.time()
        self.initialization()
        obs_n = self.env.reset()
        while True:
            action_dict = OrderedDict()
            phase_vec_dict, phase_mask_dict = self.env.get_phase_vec_mask_dict()
            int_attr_vec_dict = self.env.get_int_attr_vec_dict()

            multi_agent_obs = convert_to_tensor(data=np.array(list(obs_n.values())), data_type=torch.float32, device=self.device)
            multi_agent_phase_vec = convert_to_tensor(data=np.array(list(phase_vec_dict.values())), data_type=torch.float32, device=self.device)
            multi_agent_phase_mask = convert_to_tensor(data=np.array(list(phase_mask_dict.values())), data_type=torch.float32, device=self.device)
            multi_agent_int_attr_vec = convert_to_tensor(data=np.array(list(int_attr_vec_dict.values())), data_type=torch.float32, device=self.device)
            with torch.no_grad():
                multi_agent_policy, self.rnn_states_actor, _, _, _ = self.local_network(multi_agent_obs, multi_agent_phase_vec, multi_agent_int_attr_vec, multi_agent_phase_mask, self.rnn_states_actor)
                multi_agent_policy_dist = Categorical(multi_agent_policy)
                multi_agent_action = multi_agent_policy_dist.sample().reshape(-1)
                multi_agent_logprob_p = multi_agent_policy_dist.log_prob(multi_agent_action).reshape(-1)
            action_list = multi_agent_action.detach().cpu().clone().numpy().flatten()
            # Safety check (do not allow to select impossible phases/actions)
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

            next_obs_n, r_n, done, info = self.env.step(action_dict)

            multi_agent_rewards = list(r_n.values())
            # Store trajectory data
            self.experience_buffers[0].append(multi_agent_obs)
            self.experience_buffers[1].append(multi_agent_action)
            self.experience_buffers[2].append(multi_agent_rewards)
            self.experience_buffers[3].append(multi_agent_logprob_p)
            self.experience_buffers[4].append(multi_agent_phase_mask)
            self.experience_buffers[5].append(multi_agent_phase_vec)
            self.experience_buffers[6].append(multi_agent_int_attr_vec)

            self.episode_step += 1
            self.episode_reward += info[0]
            self.episode_action_change += info[1]

            # s = s1
            obs_n = next_obs_n

            if done:
                self.calculate_advantage_values()
                break

        run_time = time.time() - start_time
        print("{} | Reward: {}, Length: {}, Run Time: {:.2f} s".format(self.curr_episode,
                                                                       self.episode_reward,
                                                                       self.episode_step,
                                                                       run_time))

        return [self.episode_reward, self.episode_step, self.episode_action_change / self.episode_step]

    def calculate_advantage_values(self):
        """
        Calculate target values and advantages values
        Input Buffers contain:
            1. Batch observations (torch.tensor)
            2. Batch actions (torch.tensor)
            3. Batch rewards (numpy array)
            4. Batch log prob old (torch.tensor)
            5. Batch phase mask (torch.tensor)
            6. Batch phase vector (torch.tensor)
            7. Batch intersection attr vector (torch.tensor)
        (1). Output Buffers contain:
            1. Batch observations (torch.tensor)
            2. Batch actions (torch.tensor)
            3. Batch log prob old (torch.tensor)
            4. Batch phase mask (torch.tensor)
            5. Batch phase vector (torch.tensor)
            6. Batch intersection attribute vector (torch.tensor)
            7. Batch next observations for prediction (torch.tensor)
            8. Batch advantages (numpy array)
            9. Batch target values (numpy array)
        """
        gamma = NETWORK_PARAMS.GAMMA

        batch_obs = torch.stack(self.experience_buffers[0])
        batch_actions = torch.stack(self.experience_buffers[1])
        batch_log_p_old = torch.stack(self.experience_buffers[3])
        batch_phase_mask = torch.stack(self.experience_buffers[4])
        batch_phase_vec = torch.stack(self.experience_buffers[5])
        batch_int_attr_vec = torch.stack(self.experience_buffers[6])

        batch_reward = np.array(self.experience_buffers[2])

        with torch.no_grad():
            multi_agent_value, _, _, _, _ = self.local_network.forward_v(batch_obs, batch_phase_vec, batch_int_attr_vec, batch_phase_mask, None)

        adv_list, tar_v_list, bootstrap_v_list = [], [], []

        for i, tls in enumerate(self.env.tls_list):
            value_plus = multi_agent_value.clone().numpy()[:, i, :].reshape(-1)
            rewards = batch_reward[:, i]
            reward_plus = np.append(rewards.reshape(-1)[:-1], value_plus[-1])

            # Calculate target values
            target_values = calculate_discount_return(reward_plus, gamma)[:-1]

            # Calculate advantages
            deltas = reward_plus[:-1] + gamma * value_plus[1:] - value_plus[:-1]
            advantage_values = calculate_discount_return(deltas, gamma * 0.95)

            tar_v_list.append(target_values)
            adv_list.append(advantage_values)

        target_values = convert_to_tensor(data=np.stack(tar_v_list).T, data_type=torch.float32, device=self.device)
        advantage_values = convert_to_tensor(data=np.stack(adv_list).T, data_type=torch.float32, device=self.device)

        self.return_buffers[0] = batch_obs[:-1]
        self.return_buffers[1] = batch_actions[:-1]
        self.return_buffers[2] = batch_log_p_old[:-1]
        self.return_buffers[3] = batch_phase_mask[:-1]
        self.return_buffers[4] = batch_phase_vec[:-1]
        self.return_buffers[5] = batch_int_attr_vec[:-1]
        self.return_buffers[6] = batch_obs[1:]
        self.return_buffers[7] = advantage_values
        self.return_buffers[8] = target_values

        return 0

    def job(self, episode_number):
        job_results, metrics = None, None
        self.curr_episode = episode_number
        # Set the local weights to the global weights from the master network
        if TRAIN_PARAMS.LOAD_MODEL:
            weights = torch.load(TRAIN_PARAMS.EXPERIMENT_PATH + 'model/state_dict.pth', map_location=self.device)
        else:
            weights = torch.load(EXPERIMENT_PARAMS.MODEL_PATH + '/state_dict.pth', map_location=self.device)
        self.set_weights(weights=weights)

        if COMPUTE_TYPE == COMPUTE_OPTIONS.SINGLE_THREADED:
            if JOB_TYPE == JOB_OPTIONS.GET_EXPERIENCE:
                metrics = self.run_episode_single_threaded()
                job_results = [self.return_buffers, self.bootstrap_values]

            else:
                raise NotImplemented

        elif COMPUTE_TYPE == COMPUTE_OPTIONS.MULTI_THREADED:
            raise NotImplemented

        # Get the job results from the learning agents
        # and send them back to the master network
        info = {"id": self.id}

        return job_results, metrics, info


if __name__ == '__main__':
    runner = Runner(0)
    runner.run_episode_single_threaded()
