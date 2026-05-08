import argparse
from rl_env.path_env import RlGame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl
shoplistfile = 'D:/pa2/path planning2/MASAC_new1'
shoplistfile_test = 'D:/pa2/path planning2/MASAC_d_test2'
shoplistfile_test1 = 'D:/pa2/path planning2/MASAC_compare'

# Keep legacy defaults, but allow callers to override paths for ablation runs.
MODEL_PATH_LEADER = 'D:/pa2/path planning2/Path_SAC_actor_L1.pth'
MODEL_PATH_FOLLOWER = 'D:/pa2/path planning2/Path_SAC_actor_F1.pth'
TRAINING_RESULTS_PATH = shoplistfile


def configure_output_paths(model_path_leader=None, model_path_follower=None, training_results_path=None):
    """Configure save/load paths for training artifacts.

    This lets external callers (e.g., ablation entrypoints) isolate outputs
    into new folders without touching existing checkpoint files.
    """
    global MODEL_PATH_LEADER, MODEL_PATH_FOLLOWER, TRAINING_RESULTS_PATH, shoplistfile

    if model_path_leader:
        MODEL_PATH_LEADER = model_path_leader
    if model_path_follower:
        MODEL_PATH_FOLLOWER = model_path_follower
    if training_results_path:
        TRAINING_RESULTS_PATH = training_results_path
        shoplistfile = training_results_path

    for output_path in [MODEL_PATH_LEADER, MODEL_PATH_FOLLOWER, TRAINING_RESULTS_PATH]:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    print("SAC output paths configured:")
    print(f"  Leader model: {MODEL_PATH_LEADER}")
    print(f"  Follower model: {MODEL_PATH_FOLLOWER}")
    print(f"  Training results: {TRAINING_RESULTS_PATH}")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Agent=1
M_Enemy=4
RENDER=True
TRAIN_NUM = 1
TEST_EPIOSDE=100
state_number=7
action_number=None
max_action = 1.0
min_action = -1.0
EP_MAX = 500
EP_LEN = 1000
GAMMA = 0.9
q_lr = 3e-4
value_lr = 3e-3
policy_lr = 1e-3
BATCH = 256
tau = 1e-2
MemoryCapacity=50000
Switch=1
class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)
class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,outp)
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=F.relu(inputstate)
        mean=max_action*torch.tanh(self.out(inputstate))
        log_std=self.std_out(inputstate)
        log_std=torch.clamp(log_std,-20,2)
        std=log_std.exp()
        return mean,std

class CriticNet(nn.Module):
    def __init__(self,input,output):
        super(CriticNet, self).__init__()
        #q1
        self.in_to_y1=nn.Linear(input+output,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,1)
        self.out.weight.data.normal_(0,0.1)
        #q2
        self.q2_in_to_y1 = nn.Linear(input+output, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, 1)
        self.q2_out.weight.data.normal_(0, 0.1)
    def forward(self,s,a):
        inputstate = torch.cat((s, a), dim=1)
        q1=self.in_to_y1(inputstate)
        q1=F.relu(q1)
        q1=self.y1_to_y2(q1)
        q1=F.relu(q1)
        q1=self.out(q1)
        q2 = self.q2_in_to_y1(inputstate)
        q2 = F.relu(q2)
        q2 = self.q2_y1_to_y2(q2)
        q2 = F.relu(q2)
        q2 = self.q2_out(q2)
        return q1,q2

class Memory():
    def __init__(self,capacity,dims):
        self.capacity=capacity
        self.mem=np.zeros((capacity,dims))
        self.memory_counter=0
    
    def store_transition(self,s,a,r,s_):
        tran = np.hstack((s, a,r, s_))
        index = self.memory_counter % self.capacity
        self.mem[index, :] = tran
        self.memory_counter+=1
    
    def sample(self,n):
        assert self.memory_counter>=self.capacity,'Memory not full'
        sample_index = np.random.choice(self.capacity, n)
        new_mem = self.mem[sample_index, :]
        return new_mem
class Actor():
    def __init__(self, state_dim=None, action_dim=None):
        if state_dim is None:
            state_dim = state_number
        if action_dim is None:
            action_dim = action_number
            
        self.action_net=ActorNet(state_dim, action_dim)
        self.optimizer=torch.optim.Adam(self.action_net.parameters(),lr=policy_lr)

    def choose_action(self,s,evaluate=False):
        device = next(self.action_net.parameters()).device
        
        if isinstance(s, torch.Tensor):
            inputstate = s.to(device)
        else:
            inputstate = torch.FloatTensor(s).to(device)
            
        mean,std=self.action_net(inputstate)
        
        if evaluate:
            action = mean
        else:
            z = torch.randn_like(mean)
            action = torch.tanh(mean + std * z)
            
        action=torch.clamp(action,min_action,max_action)
        return action.cpu().detach().numpy()
    def evaluate(self,s, create_graph=True):
        device = next(self.action_net.parameters()).device
        
        if isinstance(s, torch.Tensor):
            inputstate = s.to(device)
        else:
            inputstate = torch.FloatTensor(s).to(device)
        
        if not create_graph:
            inputstate = inputstate.detach()
            
        mean,std=self.action_net(inputstate)
        z = torch.randn_like(mean).to(device)
        action = torch.tanh(mean + std * z)
        action = torch.clamp(action, min_action, max_action)
        
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean + std * z)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob

    def learn(self,actor_loss):
        loss=actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Entroy():
    def __init__(self):
        self.target_entropy = -2
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self,entroy_loss):
        loss=entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Critic():
    def __init__(self):
        self.critic_v,self.target_critic_v=CriticNet(state_number*(N_Agent+M_Enemy),action_number),CriticNet(state_number*(N_Agent+M_Enemy),action_number)
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr,eps=1e-5)
        self.lossfunc = nn.MSELoss()
    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self,s,a):
        return self.critic_v(s,a)

    def target_get_v(self,s,a):
        return self.target_critic_v(s,a)

    def learn(self,current_q1,current_q2,target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def main():
    parser = argparse.ArgumentParser(description='SAC Training and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Run mode: train or test')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--model_path', type=str, default='D:/pa/path planning2', 
                        help='Model save/load path')
    args = parser.parse_args()
    
    global RENDER, Switch, action_number
    
    if args.mode == 'test':
        RENDER = True
        Switch = 0
    else:
        RENDER = args.render
        Switch = 1
    
    env = RlGame(leader_count=N_Agent, follower_count=M_Enemy, obstacle_num=1, render=RENDER).unwrapped
    action_number = env.action_space.shape[0]
    run(env)

def _observation_to_array(observation):
    """Convert structured or legacy observations to a fixed NumPy array."""
    expected_agents = N_Agent + M_Enemy

    if isinstance(observation, dict):
        leader_obs = np.asarray(
            observation.get("leader", np.zeros(state_number, dtype=np.float32)),
            dtype=np.float32,
        )
        follower_obs = [
            np.asarray(item, dtype=np.float32)
            for item in observation.get("followers", [])
        ]

        all_obs = [leader_obs] + follower_obs
        if len(all_obs) < expected_agents:
            all_obs.extend(
                [np.zeros(state_number, dtype=np.float32) for _ in range(expected_agents - len(all_obs))]
            )
        elif len(all_obs) > expected_agents:
            all_obs = all_obs[:expected_agents]

        return np.stack(all_obs, axis=0)

    observation_arr = np.asarray(observation, dtype=np.float32)
    if observation_arr.ndim == 1:
        observation_arr = observation_arr.reshape(expected_agents, state_number)
    return observation_arr

def _reward_to_array(reward):
    """Convert structured or legacy rewards to a flat NumPy vector."""
    expected_agents = N_Agent + M_Enemy

    if isinstance(reward, dict):
        leader_reward = float(reward.get("leader", 0.0))
        follower_rewards = [float(item) for item in reward.get("followers", [])]
        all_rewards = [leader_reward] + follower_rewards

        if len(all_rewards) < expected_agents:
            all_rewards.extend([0.0] * (expected_agents - len(all_rewards)))
        elif len(all_rewards) > expected_agents:
            all_rewards = all_rewards[:expected_agents]

        return np.asarray(all_rewards, dtype=np.float32)

    return np.asarray(reward, dtype=np.float32).reshape(-1)

def run(env):
    if Switch == 1:
        env.set_time_step(1.0)
        print('SAC Training...')
    else:
        env.set_time_step(0.1)
        print('SAC Testing...')
        
    if Switch==1:
        all_ep_r = [[] for i in range(TRAIN_NUM)]
        all_ep_r0 = [[] for i in range(TRAIN_NUM)]
        all_ep_r1 = [[] for i in range(TRAIN_NUM)]
        for k in range(TRAIN_NUM):
            actors = [None for _ in range(N_Agent+M_Enemy)]
            critics = [None for _ in range(N_Agent+M_Enemy)]
            entroys = [None for _ in range(N_Agent+M_Enemy)]
            for i in range(N_Agent+M_Enemy):
                actors[i] = Actor()
                critics[i] = Critic()
                entroys[i] = Entroy()
            M = Memory(MemoryCapacity, 2 * state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy) + 1*(N_Agent+M_Enemy))
            ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(((N_Agent+M_Enemy), action_number)))
            action=np.zeros(((N_Agent+M_Enemy), action_number))
            for episode in range(EP_MAX):
                observation = _observation_to_array(env.reset())
                reward_totle,reward_totle0,reward_totle1 = 0,0,0
                for timestep in range(EP_LEN):
                    for i in range(N_Agent+M_Enemy):
                        action[i] = actors[i].choose_action(observation[i], evaluate=False)
                    if episode <= 20:
                        noise = ou_noise()
                    else:
                        noise = 0
                    action = action + noise
                    action = np.clip(action, -max_action, max_action)
                    observation_, reward, done, win, team_counter, dis = env.step(action)
                    observation_ = _observation_to_array(observation_)
                    reward = _reward_to_array(reward)
                    M.store_transition(observation.flatten(), action.flatten(), reward.flatten(), observation_.flatten())
                    
                    if M.memory_counter > MemoryCapacity:
                        b_M = M.sample(BATCH)
                        b_s = b_M[:, :state_number*(N_Agent+M_Enemy)]
                        b_a = b_M[:, state_number*(N_Agent+M_Enemy): state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy)]
                        b_r = b_M[:, -state_number*(N_Agent+M_Enemy) - 1*(N_Agent+M_Enemy): -state_number*(N_Agent+M_Enemy)]
                        b_s_ = b_M[:, -state_number*(N_Agent+M_Enemy):]
                        b_s = torch.FloatTensor(b_s)
                        b_a = torch.FloatTensor(b_a)
                        b_r = torch.FloatTensor(b_r)
                        b_s_ = torch.FloatTensor(b_s_)
                        
                        for i in range(N_Agent+M_Enemy):
                            new_action, log_prob_ = actors[i].evaluate(b_s_[:, state_number*i:state_number*(i+1)])
                            target_q1, target_q2 = critics[i].target_critic_v(b_s_, new_action)
                            target_q = b_r[:, i:(i+1)] + GAMMA * (torch.min(target_q1, target_q2) - entroys[i].alpha * log_prob_)
                            current_q1, current_q2 = critics[i].get_v(b_s, b_a[:, action_number*i:action_number*(i+1)])
                            critics[i].learn(current_q1, current_q2, target_q.detach())
                            a, log_prob = actors[i].evaluate(b_s[:, state_number*i:state_number*(i+1)])
                            q1, q2 = critics[i].get_v(b_s, a)
                            q = torch.min(q1, q2)
                            actor_loss = (entroys[i].alpha * log_prob - q).mean()
                            alpha_loss = -(entroys[i].log_alpha.exp() * (
                                            log_prob + entroys[i].target_entropy).detach()).mean()
                            actors[i].learn(actor_loss)
                            entroys[i].learn(alpha_loss)
                            entroys[i].alpha = entroys[i].log_alpha.exp()
                            critics[i].soft_update()
                    observation = observation_
                    reward_totle += reward.mean()
                    reward_totle0 += float(reward[0])
                    reward_totle1 += float(reward[1])
                    if RENDER:
                        env.render()
                    if done:
                        break
                print("Ep: {} rewards: {}".format(episode, reward_totle))
                leader_speed = observation[0][2] * 30
                follower_speed = observation[1][2] * 30
                print("Speed - Leader: {:.2f}, Follower: {:.2f}".format(leader_speed, follower_speed))
                all_ep_r[k].append(reward_totle)
                all_ep_r0[k].append(reward_totle0)
                all_ep_r1[k].append(reward_totle1)
                if episode % 20 == 0 and episode > 200:
                    save_data = {'net': actors[0].action_net.state_dict(), 'opt': actors[0].optimizer.state_dict()}
                    torch.save(save_data, MODEL_PATH_LEADER)
                    save_data = {'net': actors[1].action_net.state_dict(), 'opt': actors[1].optimizer.state_dict()}
                    torch.save(save_data, MODEL_PATH_FOLLOWER)
        all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
        all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
        all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
        all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
        all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
        all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
        d = {"all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std,
             "all_ep_L_mean": all_ep_L_mean, "all_ep_L_std": all_ep_L_std,
             "all_ep_F_mean": all_ep_F_mean, "all_ep_F_std": all_ep_F_std,}
        f = open(TRAINING_RESULTS_PATH, 'wb')
        pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
        f.close()
        all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
        all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
        all_ep_L_max = all_ep_L_mean + all_ep_L_std * 0.95
        all_ep_L_min = all_ep_L_mean - all_ep_L_std * 0.95
        all_ep_F_max = all_ep_F_mean + all_ep_F_std * 0.95
        all_ep_F_min = all_ep_F_mean - all_ep_F_std * 0.95
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.figure(2, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Leader reward')
        plt.figure(3, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, label='MASAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Follower reward')
        plt.legend()
        plt.show()
        env.close()
    else:
        #try:
        #    assert M_Enemy == 1
        #except:
        #    print('Error: M_Enemy must be 1 for testing')
        #    return
            
        print('SAC Testing...')
        # Use default paths for loading models
        model_path_L = MODEL_PATH_LEADER
        model_path_F = MODEL_PATH_FOLLOWER
        
        aa = Actor()
        checkpoint_aa = torch.load(model_path_L)
        aa.action_net.load_state_dict(checkpoint_aa['net'])
        
        bb = Actor()
        checkpoint_bb = torch.load(model_path_F)
        bb.action_net.load_state_dict(checkpoint_bb['net'])
        
        action = np.zeros((N_Agent+M_Enemy, action_number))
        win_times = 0
        average_FKR=0
        average_timestep=0
        average_integral_V=0
        average_integral_U= 0
        all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []
        for j in range(TEST_EPIOSDE):
            state = _observation_to_array(env.reset())
            total_rewards = 0
            integral_V=0
            integral_U=0
            v,v1,Dis=[],[],[]
            for timestep in range(EP_LEN):
                for i in range(N_Agent):
                    action[i] = aa.choose_action(state[i], evaluate=True)
                for i in range(M_Enemy):
                    action[i+1] = bb.choose_action(state[i+1], evaluate=True)
                new_state, reward, done, win, team_counter, dis = env.step(action)
                new_state = _observation_to_array(new_state)
                reward = _reward_to_array(reward)
                if win:
                    win_times += 1
                v.append(state[0][2]*30)
                v1.append(state[1][2]*30)
                Dis.append(dis)
                integral_V+=state[0][2]
                integral_U+=abs(action[0]).sum()
                total_rewards += reward.mean()
                state = new_state
                if RENDER:
                    env.render()
                if done:
                    break
            FKR=team_counter/timestep
            average_FKR += FKR
            average_timestep += timestep
            average_integral_V += integral_V
            average_integral_U += integral_U
            print("Score", total_rewards)
            avg_leader_speed = np.mean(v)
            avg_follower_speed = np.mean(v1) 
            print("Average Speed - Leader: {:.2f}, Follower: {:.2f}".format(avg_leader_speed, avg_follower_speed))
            final_leader_speed = state[0][2] * 30
            final_follower_speed = state[1][2] * 30
            print("Final Speed - Leader: {:.2f}, Follower: {:.2f}".format(final_leader_speed, final_follower_speed))
            all_ep_V.append(integral_V)
            all_ep_U.append(integral_U)
            all_ep_T.append(timestep)
            all_ep_F.append(FKR)
        print('Task completion rate:', win_times / TEST_EPIOSDE)
        print('Average formation maintenance rate:', average_FKR/TEST_EPIOSDE)
        print('Average flight time:', average_timestep/TEST_EPIOSDE)
        print('Average flight distance:', average_integral_V/TEST_EPIOSDE)
        print('Average energy consumption:', average_integral_U/TEST_EPIOSDE)
        env.close()

if __name__ == '__main__':
    main()