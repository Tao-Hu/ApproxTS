import numpy as np
import random
import copy
from scipy.stats import linregress

class AHM:

    def __init__(self, runs, mcReps, x, y, N_AM, N_YM, N_AF, T, goal, limits,
            strategy = 'TS'):
        self.runs = runs
        self.mcReps = mcReps
        self.x = copy.copy(x)
        self.y = copy.copy(y)
        self.m = 0.5246
        self.ratio = 0.897
        self.gamma_S = 0.9407
        self.gamma_R = 0.8647
        self.s_M = 0.7896
        self.s_F = 0.6886
        self.harv_rate_AM = np.array([0.1139, 0.0977, 0.0552, 0.0088])
        self.harv_rate_YM = self.harv_rate_AM * 1.5407
        self.harv_rate_AF = self.harv_rate_AM * 0.7191
        self.harv_rate_YF = self.harv_rate_AM * 1.1175
        self.surv_AM = (1.0 - self.harv_rate_AM) * self.s_M
        self.surv_YM = (1.0 - self.harv_rate_YM) * self.s_M
        self.surv_AF = (1.0 - self.harv_rate_AF) * self.s_F
        self.surv_YF = (1.0 - self.harv_rate_YF) * self.s_F
        self.beta_R = np.array([0.7166, 0.1083, -0.0373])
        self.beta_P = np.array([2.2127, 0.342])
        self.beta_P_true = np.array([2.2127, 0.342])
        self.std = np.sqrt(0.25)
        self.N_AM = N_AM
        self.N_AM_pre = N_AM
        self.N_AM_init = N_AM
        self.N_YM = N_YM
        self.N_YM_pre = N_YM
        self.N_YM_init = N_YM
        self.N_AF = N_AF
        self.N_AF_pre = N_AF
        self.N_AF_init = N_AF
        self.N_YF = (1.0 - self.m) * (N_AM + N_YM) / self.m - N_AF
        self.N_YF_pre = (1.0 - self.m) * (N_AM + N_YM) / self.m - N_AF
        self.N_YF_init = (1.0 - self.m) * (N_AM + N_YM) / self.m - N_AF
        self.N = self.N_AM + self.N_YM + self.N_AF + self.N_YF
        self.N_pre = self.N_AM + self.N_YM + self.N_AF + self.N_YF
        self.N_init = self.N_AM + self.N_YM + self.N_AF + self.N_YF
        self.P = y[-1]
        self.P_pre = y[-1]
        self.P_init = y[-1]
        self.R = (self.beta_R[0] + self.beta_R[1] * self.P +
            self.beta_R[2] * self.N)
        self.R_pre = self.R
        self.T = T
        self.goal = goal
        self.limits = copy.copy(limits)
        self.strategy = strategy
        self.policy = 0
        self.acc_harv = 0.0
        self.cur_harv = 0.0
        self.store_N = np.zeros(runs+1)
        self.store_N[0] = self.N
        self.num_bf = 50
        self.c_AM = np.zeros(self.num_bf)
        self.c_AF = np.zeros(self.num_bf)
        self.c_YM = np.zeros(self.num_bf)
        self.c_YF = np.zeros(self.num_bf)
        self.c_P = np.zeros(self.num_bf)
        self.b_AM = np.zeros(self.num_bf)
        self.b_AF = np.zeros(self.num_bf)
        self.b_YM = np.zeros(self.num_bf)
        self.b_YF = np.zeros(self.num_bf)
        self.b_P = np.zeros(self.num_bf)
        for i in range(self.num_bf):
            # range of population: 3-16
            # range of P: 2-5
            self.b_AM[i] = 13.0 * 0.335 / self.num_bf
            self.b_AF[i] = 13.0 * 0.192 / self.num_bf
            self.b_YM[i] = 13.0 * 0.302 / self.num_bf
            self.b_YF[i] = 13.0 * 0.171 / self.num_bf
            self.b_P[i] = 3.0 / self.num_bf
            self.c_AM[i] = self.b_AM[i]/2 + i*self.b_AM[i] + 3*0.335
            self.c_AF[i] = self.b_AF[i]/2 + i*self.b_AF[i] + 3*0.192
            self.c_YM[i] = self.b_YM[i]/2 + i*self.b_YM[i] + 3*0.302
            self.c_YF[i] = self.b_YF[i]/2 + i*self.b_YF[i] + 3*0.171
            self.c_P[i] = self.b_P[i]/2 + i*self.b_P[i] + 2.0
            self.b_AM[i] = np.sqrt(self.b_AM[i])
            self.b_AF[i] = np.sqrt(self.b_AF[i])
            self.b_YM[i] = np.sqrt(self.b_YM[i])
            self.b_YF[i] = np.sqrt(self.b_YF[i])
            self.b_P[i] = np.sqrt(self.b_P[i])
        self.theta = np.zeros(4*self.num_bf)

    def one_step(self, P, N_AM, N_YM, N_AF, N_YF, tmp_id):
        N = N_AM + N_YM + N_AF + N_YF
        R = self.beta_R[0] + self.beta_R[1] * P + self.beta_R[2] * N
        N_AM = (self.gamma_S * (N_AM * self.surv_AM[tmp_id] +
            N_AF * self.gamma_R * R * self.surv_YM[tmp_id] *
            self.ratio))
        N_AF = (self.gamma_S * (N_AF * (self.surv_AF[tmp_id] +
            self.gamma_R * R * self.surv_YF[tmp_id])))
        N_YM = (self.gamma_S * (N_YM * self.surv_AM[tmp_id] +
            N_YF * self.gamma_R * R * self.surv_YM[tmp_id] *
            self.ratio))
        N_YF = (self.gamma_S * (N_YF * (self.surv_AF[tmp_id] +
            self.gamma_R * R * self.surv_YF[tmp_id])))
        N = N_AM + N_AF + N_YM + N_YF
        return N

    def policy_search(self):
        acc_harv = np.zeros(4)
        #last_harv = np.zeros(3)
        for policy_id in range(4):
            for mc_id in range(self.mcReps):
                P = self.P
                N_AM = self.N_AM
                N_YM = self.N_YM
                N_AF = self.N_AF
                N_YF = self.N_YF
                N = self.N
                for i in range(self.T):
                    #tmp_N = self.one_step(P, N_AM, N_YM, N_AF, N_YF, policy_id)
                    tmp_N = N
                    # if tmp_N <= self.goal:
                    #     tmp_id = 3
                    # else:
                    #     tmp_id = policy_id
                    tmp_id = policy_id
                    acc_harv[policy_id] += self.harv_rate_AM[tmp_id] * N_AM
                    acc_harv[policy_id] += self.harv_rate_AF[tmp_id] * N_AF
                    acc_harv[policy_id] += self.harv_rate_YM[tmp_id] * N_YM
                    acc_harv[policy_id] += self.harv_rate_YF[tmp_id] * N_YF
                    R = self.beta_R[0] + self.beta_R[1] * P + self.beta_R[2] * N
                    P = (self.beta_P[0] + self.beta_P[1] * P +
                        random.gauss(0, self.std))
                    N_AM = (self.gamma_S * (N_AM * self.surv_AM[tmp_id] +
                        N_AF * self.gamma_R * R * self.surv_YM[tmp_id] *
                        self.ratio))
                    N_AF = (self.gamma_S * (N_AF * (self.surv_AF[tmp_id] +
                        self.gamma_R * R * self.surv_YF[tmp_id])))
                    N_YM = (self.gamma_S * (N_YM * self.surv_AM[tmp_id] +
                        N_YF * self.gamma_R * R * self.surv_YM[tmp_id] *
                        self.ratio))
                    N_YF = (self.gamma_S * (N_YF * (self.surv_AF[tmp_id] +
                        self.gamma_R * R * self.surv_YF[tmp_id])))
                    N = N_AM + N_AF + N_YM + N_YF

        self.policy = np.argmax(acc_harv)

    def harvest_update(self, time):
        if self.strategy == 'mixed':
            if self.N <= self.limits[0]:
                tmp_id = 3
            elif self.N <= self.limits[1]:
                tmp_id = 2
            elif self.N <= self.limits[2]:
                tmp_id = 1
            else:
                tmp_id = 0
        else:
            #tmp_N = self.one_step(self.P, self.N_AM, self.N_YM, self.N_AF,
            #    self.N_YF, self.policy)
            tmp_N = self.N
            # if tmp_N <= self.goal:
            #     tmp_id = 3
            # else:
            #     tmp_id = self.policy
            tmp_id = self.policy

        self.acc_harv += self.harv_rate_AM[tmp_id] * self.N_AM
        self.acc_harv += self.harv_rate_AF[tmp_id] * self.N_AF
        self.acc_harv += self.harv_rate_YM[tmp_id] * self.N_YM
        self.acc_harv += self.harv_rate_YF[tmp_id] * self.N_YF
        self.cur_harv = 0.0
        self.cur_harv += self.harv_rate_AM[tmp_id] * self.N_AM
        self.cur_harv += self.harv_rate_AF[tmp_id] * self.N_AF
        self.cur_harv += self.harv_rate_YM[tmp_id] * self.N_YM
        self.cur_harv += self.harv_rate_YF[tmp_id] * self.N_YF

        self.N_AM_pre = self.N_AM
        self.N_AF_pre = self.N_AF
        self.N_YM_pre = self.N_YM
        self.N_YF_pre = self.N_YF
        self.N_pre = self.N
        self.P_pre = self.P
        self.R_pre = self.R

        self.R = (self.beta_R[0] + self.beta_R[1] * self.P +
            self.beta_R[2] * self.N)
        self.x = copy.copy(np.append(self.x, self.P))
        self.P = (self.beta_P_true[0] + self.beta_P_true[1] * self.P +
            random.gauss(0, self.std))
        self.y = copy.copy(np.append(self.y, self.P))
        self.N_AM = (self.gamma_S * (self.N_AM * self.surv_AM[tmp_id] +
            self.N_AF * self.gamma_R * self.R * self.surv_YM[tmp_id] *
            self.ratio))
        self.N_AF = (self.gamma_S * (self.N_AF * (self.surv_AF[tmp_id] +
            self.gamma_R * self.R * self.surv_YF[tmp_id])))
        self.N_YM = (self.gamma_S * (self.N_YM * self.surv_AM[tmp_id] +
            self.N_YF * self.gamma_R * self.R * self.surv_YM[tmp_id] *
            self.ratio))
        self.N_YF = (self.gamma_S * (self.N_YF * (self.surv_AF[tmp_id] +
            self.gamma_R * self.R * self.surv_YF[tmp_id])))
        self.N = self.N_AM + self.N_AF + self.N_YM + self.N_YF
        self.store_N[time+1] = self.N

    def mle(self):
        self.beta_P[1], self.beta_P[0], r, p, std = linregress(self.x, self.y)

    def construct_bf(self, N_AM, N_AF, N_YM, N_YF, P, action):
        phi = np.zeros(self.num_bf)
        acc_sum = 0.0
        for i in range(self.num_bf):
            phi[i] = np.exp(- (N_AM - self.c_AM[i])**2 / (self.b_AM[i]**2)
                - (N_AF - self.c_AF[i])**2 / (self.b_AF[i]**2)
                - (N_YM - self.c_YM[i])**2 / (self.b_YM[i]**2)
                - (N_YF - self.c_YF[i])**2 / (self.b_YF[i]**2)
                - (P - self.c_P[i])**2 / (self.b_P[i]**2))
            acc_sum += phi[i]
        phi = phi/acc_sum
        if action == 0:
            out = np.concatenate((phi, np.zeros(3*self.num_bf)))
        elif action == 1:
            out = np.concatenate((np.zeros(self.num_bf), phi,
                np.zeros(2*self.num_bf)))
        elif action == 2:
            out = np.concatenate((np.zeros(2*self.num_bf), phi,
                np.zeros(self.num_bf)))
        else:
            out = np.concatenate((np.zeros(3*self.num_bf), phi))

        return out

    def approximate_dp(self, time):
        tmp_id = self.policy
        f_max = 0.0
        for mc_id in range(self.mcReps):
            R = (self.beta_R[0] + self.beta_R[1] * self.P_pre +
                self.beta_R[2] * self.N_pre)
            P = (self.beta_P_true[0] + self.beta_P_true[1] * self.P_pre +
                random.gauss(0, self.std))
            N_AM = (self.gamma_S * (self.N_AM_pre * self.surv_AM[tmp_id] +
                self.N_AF_pre * self.gamma_R * R * self.surv_YM[tmp_id] *
                self.ratio))
            N_AF = (self.gamma_S * (self.N_AF_pre * (self.surv_AF[tmp_id] +
                self.gamma_R * R * self.surv_YF[tmp_id])))
            N_YM = (self.gamma_S * (self.N_YM_pre * self.surv_AM[tmp_id] +
                self.N_YF_pre * self.gamma_R * R * self.surv_YM[tmp_id] *
                self.ratio))
            N_YF = (self.gamma_S * (self.N_YF_pre * (self.surv_AF[tmp_id] +
                self.gamma_R * R * self.surv_YF[tmp_id])))
            maximizer = np.dot(self.construct_bf(N_AM, N_AF, N_YM, N_YF, P, 0),
                self.theta)
            for i in range(3):
                tmp = np.dot(self.construct_bf(N_AM, N_AF, N_YM, N_YF, P, i+1),
                    self.theta)
                maximizer = max(maximizer, tmp)
            f_max += maximizer
            # f_max += cur_harv
        f_max = f_max / self.mcReps
        bfs = self.construct_bf(self.N_AM_pre, self.N_AF_pre, self.N_YM_pre,
            self.N_YF_pre, self.P_pre, self.policy)
        self.theta = (self.theta + (self.cur_harv + f_max
            - np.dot(bfs, self.theta)) * bfs / (time + 1))
        # self.theta = (self.theta
        #     + (f_max - np.dot(bfs, self.theta)) * bfs / (time + 1))
        maximizer = np.dot(self.construct_bf(self.N_AM, self.N_AF, self.N_YM,
            self.N_YF, self.P, 0), self.theta)
        self.policy = 0
        for i in range(3):
            tmp = np.dot(self.construct_bf(self.N_AM, self.N_AF, self.N_YM,
                self.N_YF, self.P, i+1), self.theta)
            if tmp > maximizer:
                maximizer = tmp
                self.policy = i + 1

    def move(self):
        if self.strategy == 'TS':
            self.policy_search()
            for i in range(self.runs):
                self.harvest_update(i)
                self.mle()
                self.policy_search()
        elif self.strategy == 'liberal':
            self.policy = 0
            for i in range(self.runs):
                self.harvest_update(i)
                #self.mle()
        elif self.strategy == 'moderate':
            self.policy = 1
            for i in range(self.runs):
                self.harvest_update(i)
                #self.mle()
        elif self.strategy == 'restrict':
            self.policy = 2
            for i in range(self.runs):
                self.harvest_update(i)
                #self.mle()
        elif self.strategy == 'mixed':
            for i in range(self.runs):
                self.harvest_update(i)
                #self.mle()
        elif self.strategy == 'DP':
            # self.policy = np.random.randint(0, 4)
            self.policy = 0
            for i in range(self.runs):
                self.harvest_update(i)
                self.approximate_dp(i)
                
