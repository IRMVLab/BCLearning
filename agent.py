import numpy as np
import torch
import torch.nn as nn
import pickle
import random


def xdist(pos, orn, x_n):
    return pos[0] + 0.02 * np.cos(orn[2] + np.pi / 2) - x_n


def ydist(pos, orn, y_n):
    return pos[1] + 0.02 * np.sin(orn[2] + np.pi / 2) - y_n


def da(angle, orn):
    return np.tanh(2 * (angle - orn))



def base_template(s, z1, z2, z3, K=5):
    x_n = s[0]
    y_n = s[1]
    height = s[2]
    gripper_angle = s[5]
    finger_angle = s[6]
    finger_force = s[7]
    pos1 = s[8:11]
    orn1 = s[11:14]
    pos2 = s[14:17]
    orn2 = s[17:20]
    action = np.zeros(5)
    if finger_force < 1:
        action[0] = np.tanh(xdist(pos1, orn1, x_n) * K)
        action[1] = np.tanh(ydist(pos1, orn1, y_n) * K)
        action[2] = np.tanh(K * (pos1[2] + z1 - height))
        action[3] = da(gripper_angle, orn1[2])
        action[4] = da(0, finger_angle)
        if pos1[2] + z1 < height < z2:
            action[4] = da(0.2, finger_angle)
    else:
        action[0] = np.tanh((pos2[0] - pos1[0]) * K)
        action[1] = np.tanh((pos2[1] - pos1[1]) * K)
        action[2] = np.tanh(K * (z3 - pos1[2]))
        action[3] = da(gripper_angle, orn2[2])
        action[4] = da(0, finger_angle)
        if np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) < 0.02:
            action[4] = da(0.2, finger_angle)
    return action


def base1(s):
    return base_template(s, z1=0.25, z2=0.35, z3=0.25)


def base2(s):
    return base_template(s, z1=0.25, z2=0.35, z3=0.1)


def base2_ensemble(s):
    x_n = s[0]
    y_n = s[1]
    height = s[2]
    gripper_angle = s[5]
    finger_angle = s[6]
    finger_force = s[7]
    block_pos = s[8:11]
    block_orn = s[11:14]
    block2_pos = s[14:17]
    block2_orn = s[17:20]

    def reach():
        action = np.zeros(5)
        K = 10
        if finger_force < 1:
            action[0] = np.tanh(xdist(block_pos, block_orn, x_n) * K)
            action[1] = np.tanh(ydist(block_pos, block_orn, y_n) * K)
            action[2] = np.tanh(K * (block_pos[2] + 0.22 - height))
            action[3] = da(gripper_angle, block_orn[2])
        else:
            action[0] = np.tanh((block2_pos[0] - block_pos[0]) * K)
            action[1] = np.tanh((block2_pos[1] - block_pos[1]) * K)
            action[2] = np.tanh(K * (0.1 - block_pos[2]))
            action[3] = da(gripper_angle, block2_orn[2])
        return action

    def grasp():
        action = np.zeros(5)
        if finger_force < 1:
            action[4] = da(0, finger_angle)
            if block_pos[2] + 0.25 < height < 0.35:
                action[4] = da(0.2, finger_angle)
        return action

    def drop():
        action = np.zeros(5)
        if finger_force >= 1:
            action[4] = da(0, finger_angle)
            if np.sqrt((block2_pos[0] - block_pos[0]) ** 2 + (block2_pos[1] - block_pos[1]) ** 2) < 0.02:
                action[4] = da(0.2, finger_angle)
        return action
    return reach(), grasp(), drop()


def base3(s):
    return base_template(s, z1=0.3, z2=0.45, z3=0.35)


def base_controller(s, base):
    size = s.shape[0]
    action = np.zeros((size, 5), dtype=np.float32)
    for i in range(size):
        action[i, :] = base(s[i, :])
    return action


def base_controller_ensemble(s, base):
    size = s.shape[0]
    action1 = np.zeros((size, 5), dtype=np.float32)
    action2 = np.zeros((size, 5), dtype=np.float32)
    action3 = np.zeros((size, 5), dtype=np.float32)
    for i in range(size):
        action1[i, :], action2[i, :], action3[i, :] = base(s[i, :])
    return action1, action2, action3


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t


def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.FloatTensor), device)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SpacialSoftmaxExpectation(nn.Module):
    def __init__(self, size, device):
        super(SpacialSoftmaxExpectation, self).__init__()
        cor = opt_cuda(torch.arange(size).type(torch.FloatTensor), device)
        X, Y = torch.meshgrid(cor, cor)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        self.fixed_weight = torch.cat((Y, X), dim=1)
        self.fixed_weight /= size - 1

    def forward(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).matmul(self.fixed_weight).view(x.size(0), -1)


class Actor(nn.Module):
    def __init__(self, mode, width, device):
        super(Actor, self).__init__()
        self.mode = mode
        self.width = width
        self.device = device
        self.in_channel = 4 if self.mode == 'rgbd' else 3
        self.out_channel = 16 if self.mode == 'rgbd' else 8
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(32 + 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh())

    def forward(self, x, robot_state):
        if self.mode == 'rgbd':
            x2 = self.conv(x / 255)
        else:
            x2 = torch.cat((self.conv(x[:, :3] / 255), self.conv(x[:, 3:] / 255)), dim=1)
        x3 = SpacialSoftmaxExpectation(self.width - 6, self.device)(x2)
        # concatenate with robot state:
        return self.fc(torch.cat((x3, robot_state), dim=1))  # 32 + 8


class FastActor(nn.Module):
    def __init__(self):
        super(FastActor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh())

    def forward(self, s):
        return self.fc(s)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, state, action):
        return self.fc(torch.cat((state, action), dim=1))


class ReplayBuffer:
    def __init__(self, c, w, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.obv1_buf = np.zeros([size, c, w, w], dtype=np.uint8)
        self.obv2_buf = np.zeros([size, c, w, w], dtype=np.uint8)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obv, sta, act, next_obv, next_sta, rew, done):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.obv1_buf[self.ptr] = obv
        self.obv2_buf[self.ptr] = next_obv
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    obv1=self.obv1_buf[idxs],
                    obv2=self.obv2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class ReplayBufferFast:
    def __init__(self, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sta, act, next_sta, rew, done):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def merge(self, buffer_fast):
        for i in range(buffer_fast.size):
            self.store(buffer_fast.sta1_buf[i],
                       buffer_fast.acts_buf[i],
                       buffer_fast.sta2_buf[i],
                       buffer_fast.rews_buf[i],
                       buffer_fast.done_buf[i])

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class WBAgent:
    def __init__(self, log_dir, mode='de', width=128, device=0, use_fast=False, task=1, ensemble=False,
                 mixed_q=True, base_boot=True, behavior_clone=True, imitate=False):
        self.mode = mode
        self.width = width
        self.device = device
        self.use_fast = use_fast
        self.task = task
        self.mixed_q = mixed_q
        self.base_boot = base_boot
        self.behavior_clone = behavior_clone
        self.imitate = imitate
        self.ensemble = ensemble
        if self.imitate:
            with open(log_dir + '/demo.pkl', 'rb') as fd:
                self.demo = pickle.load(fd)
        if self.task == 1:
            self.base = base1
        elif self.task == 2:
            self.base = base2_ensemble if self.ensemble else base2
        elif self.task == 3:
            self.base = base3
        if self.use_fast:
            self.buffer = ReplayBufferFast(20, 5, size=1000000)
        else:
            self.buffer = ReplayBuffer(6 if self.mode == 'de' else 4, self.width, 20, 5, size=100000)
        if self.use_fast:
            self.actor = opt_cuda(FastActor(), self.device)
            self.target_actor = opt_cuda(FastActor(), self.device)
        else:
            self.actor = opt_cuda(Actor(mode=self.mode, width=self.width, device=self.device), self.device)
            self.target_actor = opt_cuda(Actor(mode=self.mode, width=self.width, device=self.device), self.device)
        soft_update(self.target_actor, self.actor, 1)
        self.critic = opt_cuda(Critic(), self.device)
        self.target_critic = opt_cuda(Critic(), self.device)
        soft_update(self.target_critic, self.critic, 1)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1
        self.delta = 2e-5
        self.batch_size = 256

    def act(self, o, s, test=False):
        if self.ensemble:
            action_b = [a for a in self.base(s)]
        else:
            action_b = self.base(s)
        s = np_to_tensor(s, self.device).unsqueeze(dim=0)
        if self.use_fast:
            with torch.no_grad():
                action = self.actor(s)
        else:
            o = np_to_tensor(o, self.device).unsqueeze(dim=0)
            s_p = s[:, :8]
            with torch.no_grad():
                action = self.actor(o, s_p)
        if test or self.imitate:
            return action.squeeze().cpu().numpy(), True
        if np.random.uniform(0, 1) < self.epsilon:
            self.epsilon = max(self.epsilon - self.delta, 0)
            if self.ensemble:
                return random.choice(action_b), False
            else:
                return action_b, False
        else:
            self.epsilon = max(self.epsilon - self.delta, 0)
            if self.ensemble:
                action_b_t = [np_to_tensor(a, self.device).unsqueeze(dim=0) for a in action_b]
                with torch.no_grad():
                    q = self.critic(s, action)
                    max_q = q
                    for i in range(len(action_b_t)):
                        q_b = self.critic(s, action_b_t[i])
                        if q_b.item() > max_q.item():
                            best_action = action_b[i]
                            max_q = q_b
                if self.mixed_q and max_q.item() != q.item():
                    return best_action, False
                else:
                    return action.squeeze().cpu().numpy(), True
            else:
                action_b_t = np_to_tensor(action_b, self.device).unsqueeze(dim=0)
                with torch.no_grad():
                    q_b = self.critic(s, action_b_t)
                    q = self.critic(s, action)
                if q_b.item() > q.item() and self.mixed_q:
                    return action_b, False
                else:
                    return action.squeeze().cpu().numpy(), True

    def remember(self, observation, state, action, next_observation, next_state, reward, done):
        if self.use_fast:
            self.buffer.store(state, action, next_state, [reward], [done])
        else:
            self.buffer.store(observation, state, action, next_observation, next_state, [reward], [done])

    def train(self, frame):
        total_Lc = total_La = total_Lbc = 0
        steps = min(int(frame), max((5 * self.buffer.size) // self.batch_size, 1))
        for i in range(steps):
            batch = self.buffer.sample_batch(batch_size=self.batch_size)
            si = np_to_tensor(batch['sta1'], self.device)
            sn = np_to_tensor(batch['sta2'], self.device)
            ai = np_to_tensor(batch['acts'], self.device)
            ri = np_to_tensor(batch['rews'], self.device)
            d = np_to_tensor(batch['done'], self.device)
            if not self.use_fast:
                oi = np_to_tensor(batch['obv1'], self.device)
                on = np_to_tensor(batch['obv2'], self.device)
                si_p = np_to_tensor(batch['sta1'][:, :8], self.device)
                sn_p = np_to_tensor(batch['sta2'][:, :8], self.device)
            if self.imitate:
                demo_batch = self.demo.sample_batch(batch_size=int(self.batch_size/2))
                si_d = np_to_tensor(demo_batch['sta1'], self.device)
                ai_d = np_to_tensor(demo_batch['acts'], self.device)
                si = torch.cat((si, si_d), dim=0)
                sn = torch.cat((sn, np_to_tensor(demo_batch['sta2'], self.device)), dim=0)
                ai = torch.cat((ai, ai_d), dim=0)
                ri = torch.cat((ri, np_to_tensor(demo_batch['rews'], self.device)), dim=0)
                d = torch.cat((d, np_to_tensor(demo_batch['done'], self.device)), dim=0)
            if self.ensemble:
                base_action = [np_to_tensor(ba, self.device)
                               for ba in base_controller_ensemble(batch['sta1'], self.base)]
                base_action_n = [np_to_tensor(ban, self.device)
                                 for ban in base_controller_ensemble(batch['sta2'], self.base)]
            else:
                base_action = np_to_tensor(base_controller(batch['sta1'], self.base), self.device)
                base_action_n = np_to_tensor(base_controller(batch['sta2'], self.base), self.device)

            if self.imitate:
                self.optimizer_critic.zero_grad()
                with torch.no_grad():
                    yi = ri + (1 - d) * self.gamma * self.target_critic(sn, self.target_actor(sn))
                Lc = ((self.critic(si, ai) - yi) ** 2).mean()
                Lc.backward()
                self.optimizer_critic.step()
                soft_update(self.target_critic, self.critic, self.tau)
                total_Lc += Lc.item()

                self.optimizer_actor.zero_grad()
                q_a = self.critic(si, self.actor(si))
                a_d = self.actor(si_d)
                with torch.no_grad():
                    q_a_d = self.critic(si_d, a_d)
                    q_ai_d = self.critic(si_d, ai_d)
                    xi = nn.ReLU()(torch.sign(q_ai_d - q_a_d))
                Lbc = (((a_d - ai_d) ** 2).mean(dim=1, keepdim=True) * xi).sum() / max(xi.sum().item(), 1)
                La = Lbc - 0.02 * q_a.mean()
                La.backward()
                self.optimizer_actor.step()
                soft_update(self.target_actor, self.actor, self.tau)
                total_Lbc += Lbc.item()
                total_La += La.item()

            else:
                self.optimizer_critic.zero_grad()
                with torch.no_grad():
                    if self.use_fast:
                        a_next = self.target_actor(sn)
                    else:
                        a_next = self.target_actor(on, sn_p)
                    back_up = self.target_critic(sn, a_next)
                    if self.base_boot:
                        if self.ensemble:
                            back_up_d = torch.max(
                                torch.cat([self.target_critic(sn, ban) for ban in base_action_n], dim=1),
                                dim=1, keepdim=True)[0]
                        else:
                            back_up_d = self.target_critic(sn, base_action_n)
                        back_up = torch.max(back_up, back_up_d)
                    yi = ri + (1 - d) * self.gamma * back_up
                Lc = ((self.critic(si, ai) - yi) ** 2).mean()
                Lc.backward()
                self.optimizer_critic.step()
                soft_update(self.target_critic, self.critic, self.tau)
                total_Lc += Lc.item()

                self.optimizer_actor.zero_grad()
                if self.behavior_clone:
                    with torch.no_grad():
                        if self.ensemble:
                            q_ai_d, idx = torch.max(
                                torch.cat([self.critic(si, ba) for ba in base_action], dim=1),
                                dim=1, keepdim=True)
                            base_action = torch.cat([base_action[int(c)][i].unsqueeze(dim=0)
                                                     for c, i in zip(list(idx), range(idx.shape[0]))], dim=0)
                        else:
                            q_ai_d = self.critic(si, base_action)
                    if self.use_fast:
                        a = self.actor(si)
                    else:
                        a = self.actor(oi, si_p)
                    q_a = self.critic(si, a)
                    with torch.no_grad():
                        xi = nn.ReLU()(torch.sign(q_ai_d - q_a))
                    Lbc = (((a - base_action) ** 2).mean(dim=1, keepdim=True) * xi).sum() / max(xi.sum().item(), 1)
                    La = Lbc - 0.02 * q_a.mean()
                else:
                    if self.use_fast:
                        a = self.actor(si)
                    else:
                        a = self.actor(oi, si_p)
                    La = - self.critic(si, a).mean()
                La.backward()
                self.optimizer_actor.step()
                soft_update(self.target_actor, self.actor, self.tau)
                if self.behavior_clone:
                    total_Lbc += Lbc.item()
                total_La += La.item()

        return total_Lc / steps, total_La / steps, total_Lbc / steps
