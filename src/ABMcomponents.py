"""
Agent Based Modeling Framework

Three base classes are defined: Facilities, Agent and Model
"""

import numpy as np
import random
import copy
from scipy.optimize import minimize
from multiprocessing import Process, Value
import copy_reg
import types

def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _reduce_method)


# *************************************************************************** #
# define class Facilities
# *************************************************************************** #

class Facilities:
    """
    Base class for facilities

    Four types of facilities are considered
        homes: for families
        schools: for students
        factories: for workers
        public areas:  for everyone
    M is the number of facilities except homes, and N is the number of agents

    Attributes:
        coords: coordinates of facilities, (num_home + M)-by-2 matrix
        all_agents: lists of all potential agents, M-by-N matrix, values 0 or 1
        family_members: lists of family members, num_of_families-by-N matrix, 
            values 0 or 1 
        networks: social networks, M-by-N-by-N array, 0 indicates no contact
        capacity: number of agents for each facility, M vector
        cur_agents: lists of current agents, M-by-N matrix, values are 0 or 1
        ind_home: list of agents who at home, N vector, values are 0 or 1
        ind_S: lists of S agents, M-by-N matrix, values are 0 or 1
        ind_I: lists of I agents, M-by-N matrix, values are 0 or 1
        ind_R: lists of R agents, M-by-N matrix, values are 0 or 1
    """

    def __init__(
            self, num_home, num_school, num_factory, num_public, coords,
            all_agents, family_members, networks, capacity):
        num_agents = np.shape(all_agents)[1]
        self.num_home = num_home
        self.num_school = num_school
        self.num_factory = num_factory
        self.num_public = num_public
        self.coords = copy.copy(coords)
        self.all_agents = copy.copy(all_agents)
        self.family_members = copy.copy(family_members)
        self.networks = copy.copy(networks)
        self.capacity = copy.copy(capacity)
        m = num_school + num_factory + num_public
        self.cur_agents = np.zeros((m, num_agents))
        self.ind_home = np.ones(num_agents)
        self.ind_S = np.zeros((m, num_agents))
        self.ind_I = np.zeros((m, num_agents))
        self.ind_R = np.zeros((m, num_agents))

    def reproduce(self, facilities):
        self.cur_agents = copy.copy(facilities.cur_agents)
        self.ind_home = copy.copy(facilities.ind_home)
        self.ind_S = copy.copy(facilities.ind_S)
        self.ind_I = copy.copy(facilities.ind_I)
        self.ind_R = copy.copy(facilities.ind_R)
        


# *************************************************************************** #
# define class Agent
# *************************************************************************** #

class Agent:
    """
    Base class for agents

    Attributes
        time: system time, unit is day
        identity: positive integer, agent's ID
        family_id: positive integer, agent's family ID
        location: positive integer, identifier of location where the agent is
        gender: 0 for female and 1 for male 
        age: positive value
        health_status: 0 for S, 1 for I, and 2 for R
        social_type: 0 for students, 1 for workers and 2 for retired
        ind_infected: infected indicator, 0 for not infected and 1 for infected
        ind_trt: 0 for not applying treatment and 1 otherwise
        p_out: probability of going out even if infected
        cov: covariates such as age, gender, social type
    """

    def __init__(
            self, runs, identity, family_id, location, gender, age, health_status, 
            social_type, ind_trt, p_out, susceptibility, zeta, beta, facilities):
        self.time = 0
        self.identity = identity
        self.family_id = family_id
        self.location = location
        self.gender = gender
        self.age = age / 90.0
        self.health_status = health_status
        self.health_status_previous = health_status
        self.social_type = social_type
        self.ind_trt = np.zeros(runs + 1)
        self.ind_trt[0] = ind_trt
        self.p_out = p_out
        self.susceptibility = susceptibility
        self.zeta = zeta
        self.beta = beta
        self.ind_infected = np.zeros(runs+1)
        if self.health_status == 1:
            self.ind_infected[0] = 1.0
            self.ind_infected_previous = 1.0
        else:
            self.ind_infected[0] = 0.0
            self.ind_infected_previous = 0.0
        # identify destinations for an agent
        num_home = facilities.num_home
        num_school = facilities.num_school
        num_factory = facilities.num_factory
        num_public = facilities.num_public
        self.destn = np.flatnonzero(facilities.all_agents[:, self.identity])[0]
        # feature vector: age, susceptibility, connectivity
        baseVec = np.array([[age / 90.0, susceptibility, 0.0]])
        self.cov = np.repeat(baseVec, runs+1, axis=0)

    def reproduce(self, agent):
        self.time = agent.time
        self.location = agent.location
        self.health_status = agent.health_status
        self.health_status_previous = agent.health_status_previous
        self.ind_trt = copy.copy(agent.ind_trt)
        self.ind_infected = copy.copy(agent.ind_infected)
        self.ind_infected_previous = agent.ind_infected_previous
        self.susceptibility = agent.susceptibility
        self.cov = copy.copy(agent.cov)

    def move(self, facilities):
        num_home = facilities.num_home
        num_school = facilities.num_school
        num_factory = facilities.num_factory
        num_public = facilities.num_public
        m = num_school + num_factory + num_public
        time = self.time
        if self.location < m:                # not at home
            facilities.cur_agents[self.location, self.identity] = 0
            if self.health_status_previous == 0:
                facilities.ind_S[self.location, self.identity] = 0
            elif self.health_status_previous == 1:
                facilities.ind_I[self.location, self.identity] = 0
            else:
                facilities.ind_R[self.location, self.identity] = 0
        else:                                # at home
            facilities.ind_home[self.identity] = 0
        # stay at home
        if self.ind_infected[time] == 1 and np.random.random() > self.p_out:
            self.location = m + self.family_id
            facilities.ind_home[self.identity] = 1
        # go out
        else:
            self.location = self.destn
            facilities.cur_agents[self.location, self.identity] = 1
            if self.health_status == 0:
                facilities.ind_S[self.location, self.identity] = 1
            elif self.health_status == 1:
                facilities.ind_I[self.location, self.identity] = 1
            else:
                facilities.ind_R[self.location, self.identity] = 1
        


# *************************************************************************** #
# define class Model
# *************************************************************************** #

class Model:
    """
    Base class for models

    Attributes
        num_agents: number of agents/homes, positive integer
        num_school: number of schools, positive integer
        num_factory: number of factories, positive integer
        num_public: number of public areas, positive integer
        agents: dictionary, list of all agents
        facilities: dictionary, instance of Facilities class
        p_StoI: probability of S status to I status
        p_ItoR: probability of I status to R status
        p_RtoS: probability of R status to S status
        eta: dictionary, parameters for deterministic policy
        num_trt_per_iter: number of treatments applied during each iteration
        theta: parameters for system dynamics
        # func_name: index log-likelihood function and corresponding derivative
        # loglik_dict: dictionary, store log-likelihood functions
        # deriv_dict:dictionary, store derivative functions
        contact_dict: dictionary type, store subset of infectious agents who 
                      have direct contact with susceptible agents
        mcReps: number of MC repliocates to approximate truncated value function
        tau: parameter for simultaneous perturbation
        rho: parameter for simultaneous perturbation
        l: parameter for simultaneous perturbation
        C: parameter for simultaneous perturbation
        t: parameter for simultaneous perturbation
        tol: parameter for simultaneous perturbation
        T: parameter for simultaneous perturbation
        strategy: strings, indicates for strategy, "TS" or "none"
    """

    # ----------------------------------------------------------------------- #
    # class initial
    # ----------------------------------------------------------------------- #

    def __init__(
            self, runs, n, num_home, num_school, num_factory, num_public, 
            p_StoI, p_ItoR, p_RtoS, coords, all_agents, family_members, 
            networks, capacity, identity, family_id, location, gender, age, 
            health_status, social_type, ind_trt, p_out, susceptibility, 
            zeta, beta, eta, num_trt_per_iter, theta, 
            mcReps, tau, rho, l, C, t, tol, T, strategy = 'TS'):
        self.runs = runs
        self.num_agents = n
        self.num_home = num_home
        self.num_school = num_school
        self.num_factory = num_factory
        self.num_public = num_public
        self.facilities = {}
        self.facilities['0'] = Facilities(num_home, num_school, num_factory, 
            num_public, coords, all_agents, family_members, networks, capacity)
        self.facilities['1'] = Facilities(num_home, num_school, num_factory, 
            num_public, coords, all_agents, family_members, networks, capacity)
        self.facilities['2'] = Facilities(num_home, num_school, num_factory, 
            num_public, coords, all_agents, family_members, networks, capacity)
        self.facilitiesInit = Facilities(num_home, num_school, num_factory, 
            num_public, coords, all_agents, family_members, networks, capacity)
        self.agents = {}
        self.agentsInit = []
        self.create_agents(runs+T+1, identity, family_id, location, gender, 
            age, health_status, social_type, ind_trt, p_out, susceptibility, 
            zeta, beta)
        self.zeta = zeta
        self.beta = beta
        self.p_StoI = p_StoI
        self.p_ItoR = p_ItoR
        self.p_RtoS = p_RtoS
        self.summary = np.zeros(3)
        tmp_list = [self.agents['0'][i].health_status for i in range(n)]
        self.summary[0] = tmp_list.count(0) / float(n)
        self.summary[1] = tmp_list.count(1) / float(n)
        self.summary[2] = 1.0 - self.summary[0] - self.summary[1]
        self.eta = {}
        self.eta['0'] = copy.copy(eta)
        self.eta['1'] = copy.copy(eta)
        self.eta['2'] = copy.copy(eta)
        self.eta_init = copy.copy(eta)
        self.num_trt_per_iter = num_trt_per_iter
        self.theta = copy.copy(theta)
        self.theta_true = copy.copy(theta)
        self.contact_dict = {}
        self.mcReps = mcReps
        self.tau = tau
        self.rho = rho
        self.l = l
        self.C = C
        self.t = t
        self.tol = tol
        self.T = T
        self.strategy = strategy

    def create_agents(
            self, runs, identity, family_id, location, gender, age, 
            health_status, social_type, ind_trt, p_out, susceptibility, zeta, 
            beta):
        self.agents['0'] = []
        self.agents['1'] = []
        self.agents['2'] = []
        for i in range(self.num_agents):
            tmp_agent = Agent(runs, identity[i], family_id[i], location[i], 
                gender[i], age[i], health_status[i], social_type[i], 
                ind_trt[i], p_out, susceptibility[i], zeta, beta, 
                self.facilities['0'])
            self.agents['0'].append(copy.deepcopy(tmp_agent))
            self.agents['1'].append(copy.deepcopy(tmp_agent))
            self.agents['2'].append(copy.deepcopy(tmp_agent))
            self.agentsInit.append(copy.deepcopy(tmp_agent))

    # ----------------------------------------------------------------------- #
    # Approximate Thompson sampling: three-step iteration
    # ----------------------------------------------------------------------- #

    def transmission(self):
        # apply current policy, generate actions and outcomes
        self.policy_apply('0')
        # MLE of system dynamic parameters
        self.mle()
        # stochastic approximation, find optimal policy
        self.sim_perb()

    # ----------------------------------------------------------------------- #
    # define policy_apply
    # ----------------------------------------------------------------------- #

    def feature_extract(self, agent_id, mode):
        """
        extract features for each agent, required by deterministic policy
        """
        
        # set time point
        time = self.agents[mode][agent_id].time

        # first store the ind_infected for previous time step
        self.agents[mode][agent_id].ind_infected_previous = (
            self.agents[mode][agent_id].ind_infected[time-1])
            
        # set susceptibility
        self.agents[mode][agent_id].cov[time, 1] = (
            self.agents[mode][agent_id].susceptibility)

        # set connectivity
        if self.agents[mode][agent_id].health_status == 2:
            self.agents[mode][agent_id].cov[time, 2] = 0.0
        else:
            ind_health = 1 - self.agents[mode][agent_id].health_status
            set_family = set(
                np.flatnonzero(self.facilities[mode].family_members[
                    self.agents[mode][agent_id].family_id, :]))
            set_family.remove(agent_id)
            set_temp = copy.copy(set_family)
            set_family = set()
            for i in set_temp:
                if self.agents[mode][agent_id].health_status == ind_health:
                    set_family.add(i)
            self.agents[mode][agent_id].cov[time, 2] = (len(set_family) * 1.0 / 
                self.num_agents)
            if self.facilities[mode].ind_home[agent_id] == 0:
                place_id = self.agents[mode][agent_id].location
                set_cur = set(np.flatnonzero(
                    self.facilities[mode].cur_agents[place_id, :]))
                if ind_health == 0:
                    set_all_contact = set(np.flatnonzero(
                        self.facilities[mode].ind_S[place_id, :]))
                else:
                    set_all_contact = set(np.flatnonzero(
                        self.facilities[mode].ind_I[place_id, :]))
                set_contact = set_all_contact & set_cur
                self.agents[mode][agent_id].cov[time, 2] = (
                    len(set_contact | set_family) * 1.0 / self.num_agents)

        # construct feature vector
        result = np.append(self.agents[mode][agent_id].cov[time, :], 
            self.agents[mode][agent_id].ind_infected[time-1])
        return result

    def tran_prob(self, agent_from, agent_to, mode):
        """
        the probability that disease will transfer from agent_from to agent_to
        """

        time = self.agents[mode][agent_from].time
        A_to = self.agents[mode][agent_to].ind_trt[time]
        A_from = self.agents[mode][agent_from].ind_trt[time]
        age_to = self.agents[mode][agent_to].cov[time, 0]
        age_from = self.agents[mode][agent_from].cov[time, 0]
        
        vec_cov = np.hstack((1.0, self.agents[mode][agent_to].cov[time, 0:2], 
            A_to, A_from, A_to * A_from, np.square(age_to - age_from)))
        if mode == '0':
            dot_prod = np.dot(vec_cov, self.theta_true)
        else:
            dot_prod = np.dot(vec_cov, self.theta)
        if dot_prod >= 40.0:
            return 1.0
        else:
            tmp_prob = np.exp(dot_prod)
            result = tmp_prob / (1.0 + tmp_prob)
            return result

    def feature_extract_iter(self, agent_id, mode):
        features = self.feature_extract(agent_id, mode)
        return np.dot(features, self.eta[mode])

    def policy_apply(self, mode):
        """
        apply current policy, generate actions and outcomes
        """

        # set time point
        time = self.agents[mode][0].time
        
        ## apply current policy to generate actions
        # priority_scores = np.zeros(self.num_agents)

        priority_scores = map(
            lambda agent_id: self.feature_extract_iter(agent_id, mode), 
            xrange(self.num_agents))

        # ind_trt = np.argsort(-priority_scores)[:self.num_trt_per_iter]
        ind_trt = np.argsort(priority_scores)[-self.num_trt_per_iter:]
        for agent_id in range(self.num_agents):
            if agent_id in ind_trt:
                self.agents[mode][agent_id].ind_trt[time] = 1.0
                self.agents[mode][agent_id].susceptibility = (self.zeta + 
                    self.beta * self.agents[mode][agent_id].susceptibility + 
                    np.random.normal(0, 0.5))
            else:
                self.agents[mode][agent_id].ind_trt[time] = 0.0
                self.agents[mode][agent_id].susceptibility = (
                    self.beta * self.agents[mode][agent_id].susceptibility + 
                    np.random.normal(0, 0.5))
        
        ## generate outcomes
        if mode == '0':
            self.contact_dict[str(time)] = {}
        # set of agents not at home
        for place_id in range(
                self.num_school + self.num_factory + self.num_public):
            set_S = set(np.flatnonzero(self.facilities[mode].ind_S[place_id, :]))
            set_I = set(np.flatnonzero(self.facilities[mode].ind_I[place_id, :]))
            set_R = set(np.flatnonzero(self.facilities[mode].ind_R[place_id, :]))

            # set of Susceptible agents
            for i_S in set_S:
                set_all_contact = set(np.flatnonzero(
                    self.facilities[mode].networks[place_id, i_S, :]))
                set_contact = set_all_contact & set_I
                p_infected = 1.0
                for i_set in set_contact:
                    p_StoI = self.tran_prob(i_set, i_S, mode)
                    p_infected = p_infected * (1.0 - p_StoI)
                set_family = set(np.flatnonzero(
                    self.facilities[mode].family_members[
                    self.agents[mode][i_S].family_id, :]))
                set_family.remove(i_S)
                set_extra = set()
                for i_set in set_family:
                    if self.agents[mode][i_set].ind_infected_previous == 1:
                        if i_set not in set_contact:
                            p_StoI = self.tran_prob(i_set, i_S, mode)
                            p_infected = p_infected * (1.0 - p_StoI)
                            set_extra.add(i_set)
                p_infected = 1.0 - p_infected
                self.agents[mode][i_S].health_status_previous = (
                    self.agents[mode][i_S].health_status)
                if np.random.random() <= p_infected:
                    self.agents[mode][i_S].health_status = 1
                    self.agents[mode][i_S].ind_infected[time] = 1.0
                # store contact subset, preparation for log-likelihood function
                if mode == '0':
                    self.contact_dict[str(time)][str(i_S)] = set_contact | set_extra

            # set of Infected agents
            for i_I in set_I:
                self.agents[mode][i_I].health_status_previous = (
                    self.agents[mode][i_I].health_status)
                if np.random.random() <= self.p_ItoR:
                    self.agents[mode][i_I].health_status = 2
                    self.agents[mode][i_I].ind_infected[time] = 0.0

            # set of Recovered agents
            for i_R in set_R:
                self.agents[mode][i_R].health_status_previous = (
                    self.agents[mode][i_R].health_status)
                if np.random.random() <= self.p_RtoS:
                    self.agents[mode][i_R].health_status = 0
                    self.agents[mode][i_R].ind_infected[time] = 0.0

        # set of infected agents at home
        for home_id in range(self.num_agents):
            if self.facilities[mode].ind_home[home_id] == 1:
                self.agents[mode][home_id].health_status_previous = (
                    self.agents[mode][home_id].health_status)
                if np.random.random() <= self.p_ItoR:
                    self.agents[mode][home_id].health_status = 2
                    self.agents[mode][home_id].ind_infected[time] = 0.0

    # ----------------------------------------------------------------------- #
    # define mle estimations
    # ----------------------------------------------------------------------- #

    # define objective function
    def obj_func(self, theta):
        func_min = 0.0
        for time in range(1, self.agents['0'][0].time + 1):
            key_list = self.contact_dict[str(time)].keys()
            for i_S in range(self.num_agents):
                if str(i_S) in key_list:
                    tmp = 0.0
                    for i_set in self.contact_dict[str(time)][str(i_S)]:
                        A_to = self.agents['0'][i_S].ind_trt[time]
                        A_from = self.agents['0'][i_set].ind_trt[time]
                        age_to = self.agents['0'][i_S].cov[time, 0]
                        age_from = self.agents['0'][i_set].cov[time, 0]
                        vec_cov = np.hstack((1.0, 
                        	self.agents['0'][i_S].cov[time, 0:2], 
                        	A_to, A_from, A_to * A_from, 
                        	np.square(age_to - age_from)))
                        dot_prod = np.dot(vec_cov, theta)
                        tmp += np.log(np.exp(dot_prod) + 1.0)
                    if self.agents['0'][i_S].ind_infected[time] == 1:
                        if tmp != 0.0:
                            tmp = np.log(np.exp(tmp) / (np.exp(tmp) - 1.0))
                    func_min += tmp
        return func_min

    # define derivative function
    def obj_der(self, theta):
        der = np.zeros_like(theta)
        sub_length = len(self.agents['0'][0].cov[0, :]) - 1
        for time in range(1, self.agents['0'][0].time + 1):
            key_list = self.contact_dict[str(time)].keys()
            for i_S in range(self.num_agents):
                if str(i_S) in key_list:
                    tmp = np.zeros_like(theta)
                    tmp3 = 0.0
                    for i_set in self.contact_dict[str(time)][str(i_S)]:
                        A_to = self.agents['0'][i_S].ind_trt[time]
                        A_from = self.agents['0'][i_set].ind_trt[time]
                        age_to = self.agents['0'][i_S].cov[time, 0]
                        age_from = self.agents['0'][i_set].cov[time, 0]
                        vec_cov = np.hstack((1.0, 
                        	self.agents['0'][i_S].cov[time, 0:2], 
                        	A_to, A_from, A_to * A_from, 
                        	np.square(age_to - age_from)))
                        dot_prod = np.dot(vec_cov, theta)
                        tmp2 = np.exp(dot_prod)
                        tmp3 += np.log(tmp2 + 1.0)
                        tmp2 = tmp2 / (tmp2 + 1.0)
                        tmp[0] += tmp2
                        for i in range(sub_length):
                            tmp[1+i] += (tmp2 * self.agents['0'][i_S].cov[time, i])
                        tmp[1+sub_length] += (tmp2 * A_to)
                        tmp[2+sub_length] += (tmp2 * A_from)
                        tmp[3+sub_length] += (tmp2 * A_to * A_from)
                        tmp[4+sub_length] += (tmp2 * np.square(age_to - age_from))
                    if self.agents['0'][i_S].ind_infected[time] == 1:
                        if tmp3 != 0.0:
                            tmp /= (1.0 - np.exp(tmp3))
                    der += tmp

        return der

    def mle(self):
        """
        MLE for the system dynamics parameters
        """

        # obtain MLE using scipy.optimize.minimize
        res = minimize(self.obj_func, self.theta, method='BFGS', 
            jac=self.obj_der, options={'disp': False})
        self.theta = copy.copy(res.x)

    # ----------------------------------------------------------------------- #
    # define simultaneous perturbation
    # ----------------------------------------------------------------------- #

    def iterator_SP(self, mcID, Val1, Val2):
        n = self.num_agents
        self.facilities['1'].reproduce(self.facilitiesInit)
        self.facilities['2'].reproduce(self.facilitiesInit)
        for i in range(n):
            self.agents['1'][i].reproduce(self.agentsInit[i])
            self.agents['1'][i].move(self.facilities['1'])
            self.agents['1'][i].time += 1
            self.agents['2'][i].reproduce(self.agentsInit[i])
            self.agents['2'][i].move(self.facilities['2'])
            self.agents['2'][i].time += 1

        tmpVal1 = 0.0
        tmpVal2 = 0.0

        # run for one MC
        for m in range(self.T):
            self.policy_apply('1')
            tmp_list = [self.agents['1'][j].health_status for j in range(n)]
            tmpVal1 += tmp_list.count(1) / float(n)
            for j in range(n):
                self.agents['1'][j].move(self.facilities['1'])
                self.agents['1'][j].time += 1

            self.policy_apply('2')
            tmp_list = [self.agents['2'][j].health_status for j in range(n)]
            tmpVal2 += tmp_list.count(1) / float(n)
            for j in range(n):
                self.agents['2'][j].move(self.facilities['2'])
                self.agents['2'][j].time += 1

        tmp1 = (tmpVal1 / self.T)
        tmp2 = (tmpVal2 / self.T)
        Val1.value += tmp1
        Val2.value += tmp2

    def sim_perb(self):
        """
        Stochastic approximation for maximization of value function
        """

        k = 1
        alpha = self.tau / (self.rho + k) ** self.l
        len_eta = len(self.eta['0'])
        z = np.ones(len_eta)
        eta_init = copy.copy(self.eta_init)
        n = self.num_agents

        while alpha > self.tol:
            # parameter perturbation
            zeta = self.C / k ** self.t
            for i in range(len_eta):
                if np.random.random() > 0.5:
                    z[i] = 1.0
                    self.eta['1'][i] = eta_init[i] + zeta
                    self.eta['2'][i] = eta_init[i] - zeta
                else:
                    z[i] = -1.0
                    self.eta['1'][i] = eta_init[i] - zeta
                    self.eta['2'][i] = eta_init[i] + zeta

            # Val1 = Value('d', 0.0)
            Val1 = 0.0
            # Val2 = Value('d', 0.0)
            Val2 = 0.0

            for mcID in range(self.mcReps):
                # initial agents and facilities
                self.facilities['1'].reproduce(self.facilitiesInit)
                self.facilities['2'].reproduce(self.facilitiesInit)
                for i in range(n):
                    self.agents['1'][i].reproduce(self.agentsInit[i])
                    self.agents['1'][i].move(self.facilities['1'])
                    self.agents['1'][i].time += 1
                    self.agents['2'][i].reproduce(self.agentsInit[i])
                    self.agents['2'][i].move(self.facilities['2'])
                    self.agents['2'][i].time += 1
            
                tmpVal1 = 0.0
                tmpVal2 = 0.0
            
                # run for one MC
                for m in range(self.T):
                    self.policy_apply('1')
                    tmp_list = [self.agents['1'][j].health_status for j in range(n)]
                    tmpVal1 += tmp_list.count(1) / float(n)
                    for j in range(n):
                        self.agents['1'][j].move(self.facilities['1'])
                        self.agents['1'][j].time += 1
            
                    self.policy_apply('2')
                    tmp_list = [self.agents['2'][j].health_status for j in range(n)]
                    tmpVal2 += tmp_list.count(1) / float(n)
                    for j in range(n):
                        self.agents['2'][j].move(self.facilities['2'])
                        self.agents['2'][j].time += 1
            
                Val1 += (tmpVal1 / self.T)
                Val2 += (tmpVal2 / self.T)

            # updates
            tmp = (Val1 - Val2) / self.mcReps
            coef = alpha / (2.0 * zeta)
            eta_init -= (coef * tmp * z)
            k += 1
            alpha = self.tau / (self.rho + k) ** self.l

        self.eta['0'] = copy.copy(eta_init)

    def step(self, mode):
        for i in range(self.num_agents):
            self.agents[mode][i].move(self.facilities[mode])
            self.agents[mode][i].time += 1
        if mode == '0':
            self.transmission()

    # ----------------------------------------------------------------------- #
    # strategy none
    # ----------------------------------------------------------------------- #

    def transmission_none(self):
        # set time point
        time = self.agents['0'][0].time

        # update cov and susceptibility
        for agent_id in range(self.num_agents):
            features = self.feature_extract(agent_id, '0')
            self.agents['0'][agent_id].susceptibility = ( 
                self.beta * self.agents['0'][agent_id].susceptibility + 
                np.random.normal(0, 0.5))

        # set of agents not at home
        for place_id in range(
                self.num_school + self.num_factory + self.num_public):
            set_S = set(np.flatnonzero(self.facilities['0'].ind_S[place_id, :]))
            set_I = set(np.flatnonzero(self.facilities['0'].ind_I[place_id, :]))
            set_R = set(np.flatnonzero(self.facilities['0'].ind_R[place_id, :]))

            # set of Susceptible agents
            for i_S in set_S:
                set_all_contact = set(np.flatnonzero(
                    self.facilities['0'].networks[place_id, i_S, :]))
                set_contact = set_all_contact & set_I
                p_infected = 1.0
                for i_set in set_contact:
                    p_StoI = self.tran_prob(i_set, i_S, '0')
                    p_infected = p_infected * (1.0 - p_StoI)
                set_family = set(np.flatnonzero(
                    self.facilities['0'].family_members[
                    self.agents['0'][i_S].family_id, :]))
                set_family.remove(i_S)
                for i_set in set_family:
                    if self.agents['0'][i_set].ind_infected_previous == 1:
                        if i_set not in set_contact:
                            p_StoI = self.tran_prob(i_set, i_S, '0')
                            p_infected = p_infected * (1.0 - p_StoI)
                p_infected = 1.0 - p_infected
                self.agents['0'][i_S].health_status_previous = (
                    self.agents['0'][i_S].health_status)
                if np.random.random() <= p_infected:
                    self.agents['0'][i_S].health_status = 1
                    self.agents['0'][i_S].ind_infected[time] = 1.0

            # set of Infected agents
            for i_I in set_I:
                self.agents['0'][i_I].health_status_previous = (
                    self.agents['0'][i_I].health_status)
                if np.random.random() <= self.p_ItoR:
                    self.agents['0'][i_I].health_status = 2
                    self.agents['0'][i_I].ind_infected[time] = 0.0

            # set of Recovered agents
            for i_R in set_R:
                self.agents['0'][i_R].health_status_previous = (
                    self.agents['0'][i_R].health_status)
                if np.random.random() <= self.p_RtoS:
                    self.agents['0'][i_R].health_status = 0
                    self.agents['0'][i_R].ind_infected[time] = 0.0

        # set of infected agents at home
        for home_id in range(self.num_agents):
            if self.facilities['0'].ind_home[home_id] == 1:
                self.agents['0'][home_id].health_status_previous = (
                    self.agents['0'][home_id].health_status)
                if np.random.random() <= self.p_ItoR:
                    self.agents['0'][home_id].health_status = 2
                    self.agents['0'][home_id].ind_infected[time] = 0.0

    def step_none(self):
        for i in range(self.num_agents):
            self.agents['0'][i].move(self.facilities['0'])
            self.agents['0'][i].time += 1
        self.transmission_none()

    # ----------------------------------------------------------------------- #
    # strategy proximal
    # ----------------------------------------------------------------------- #

    def transmission_proximal(self):
        # set time point
        time = self.agents['0'][0].time

        ## apply proximal policy to generate actions
        num_neighbor_S = np.zeros(self.num_agents)
        num_neighbor_I = np.zeros(self.num_agents)

        for agent_id in range(self.num_agents):
            features = self.feature_extract(agent_id, '0')
            if self.agents['0'][agent_id].health_status == 0:
                num_neighbor_S[agent_id] = features[2]
            elif self.agents['0'][agent_id].health_status == 1:
                num_neighbor_I[agent_id] = features[2]

        tmp = int(np.ceil(self.num_trt_per_iter / 2.0))
        n1 = min(tmp, np.count_nonzero(num_neighbor_S))
        n2 = min(tmp, np.count_nonzero(num_neighbor_I))
        ind_trt_S = np.argsort(-num_neighbor_S)[:n1]
        ind_trt_I = np.argsort(-num_neighbor_I)[:n2]

        for agent_id in range(self.num_agents):
            if agent_id in ind_trt_S:
                self.agents['0'][agent_id].ind_trt[time] = 1.0
                self.agents['0'][agent_id].susceptibility = (self.zeta + 
                    self.beta * self.agents['0'][agent_id].susceptibility + 
                    np.random.normal(0, 0.5))
            elif agent_id in ind_trt_I:
                self.agents['0'][agent_id].ind_trt[time] = 1.0
                self.agents['0'][agent_id].susceptibility = (self.zeta + 
                    self.beta * self.agents['0'][agent_id].susceptibility + 
                    np.random.normal(0, 0.5))
            else:
                self.agents['0'][agent_id].ind_trt[time] = 0.0
                self.agents['0'][agent_id].susceptibility = ( 
                    self.beta * self.agents['0'][agent_id].susceptibility + 
                    np.random.normal(0, 0.5))

        # set of agents not at home
        for place_id in range(
                self.num_school + self.num_factory + self.num_public):
            set_S = set(np.flatnonzero(self.facilities['0'].ind_S[place_id, :]))
            set_I = set(np.flatnonzero(self.facilities['0'].ind_I[place_id, :]))
            set_R = set(np.flatnonzero(self.facilities['0'].ind_R[place_id, :]))

            # set of Susceptible agents
            for i_S in set_S:
                set_all_contact = set(np.flatnonzero(
                    self.facilities['0'].networks[place_id, i_S, :]))
                set_contact = set_all_contact & set_I
                p_infected = 1.0
                for i_set in set_contact:
                    p_StoI = self.tran_prob(i_set, i_S, '0')
                    p_infected = p_infected * (1.0 - p_StoI)
                set_family = set(np.flatnonzero(
                    self.facilities['0'].family_members[
                    self.agents['0'][i_S].family_id, :]))
                set_family.remove(i_S)
                for i_set in set_family:
                    if self.agents['0'][i_set].ind_infected_previous == 1:
                        if i_set not in set_contact:
                            p_StoI = self.tran_prob(i_set, i_S, '0')
                            p_infected = p_infected * (1.0 - p_StoI)
                p_infected = 1.0 - p_infected
                self.agents['0'][i_S].health_status_previous = (
                    self.agents['0'][i_S].health_status)
                if np.random.random() <= p_infected:
                    self.agents['0'][i_S].health_status = 1
                    self.agents['0'][i_S].ind_infected[time] = 1.0

            # set of Infected agents
            for i_I in set_I:
                self.agents['0'][i_I].health_status_previous = (
                    self.agents['0'][i_I].health_status)
                if np.random.random() <= self.p_ItoR:
                    self.agents['0'][i_I].health_status = 2
                    self.agents['0'][i_I].ind_infected[time] = 0.0

            # set of Recovered agents
            for i_R in set_R:
                self.agents['0'][i_R].health_status_previous = (
                    self.agents['0'][i_R].health_status)
                if np.random.random() <= self.p_RtoS:
                    self.agents['0'][i_R].health_status = 0
                    self.agents['0'][i_R].ind_infected[time] = 0.0

        # set of infected agents at home
        for home_id in range(self.num_agents):
            if self.facilities['0'].ind_home[home_id] == 1:
                self.agents['0'][home_id].health_status_previous = (
                    self.agents['0'][home_id].health_status)
                if np.random.random() <= self.p_ItoR:
                    self.agents['0'][home_id].health_status = 2
                    self.agents['0'][home_id].ind_infected[time] = 0.0

    def step_proximal(self):
        for i in range(self.num_agents):
            self.agents['0'][i].move(self.facilities['0'])
            self.agents['0'][i].time += 1
        self.transmission_proximal()

    # ----------------------------------------------------------------------- #
    # define interface to run the Model
    # ----------------------------------------------------------------------- #

    def runner_TS(self, i, n):
        self.step('0')
        tmp_list = [self.agents['0'][j].health_status for j in range(n)]
        self.summary[0] += tmp_list.count(0) / float(n)
        self.summary[1] += tmp_list.count(1) / float(n)
        self.summary[2] += tmp_list.count(2) / float(n)

    def run_model(self):
        n = self.num_agents
        if self.strategy == 'TS':
            self.sim_perb()
            map(lambda i: self.runner_TS(i, n), xrange(self.runs))
            self.summary[0] /= (self.runs + 1)
            self.summary[1] /= (self.runs + 1)
            self.summary[2] /= (self.runs + 1)
        elif self.strategy == 'none':
            for i in range(self.runs):
                self.step_none()
                tmp_list = [self.agents['0'][j].health_status for j in range(n)]
                self.summary[0] += tmp_list.count(0) / float(n)
                self.summary[1] += tmp_list.count(1) / float(n)
                self.summary[2] += tmp_list.count(2) / float(n)
            self.summary[0] /= (self.runs + 1)
            self.summary[1] /= (self.runs + 1)
            self.summary[2] /= (self.runs + 1)
        else:
            for i in range(self.runs):
                self.step_proximal()
                tmp_list = [self.agents['0'][j].health_status for j in range(n)]
                self.summary[0] += tmp_list.count(0) / float(n)
                self.summary[1] += tmp_list.count(1) / float(n)
                self.summary[2] += tmp_list.count(2) / float(n)
            self.summary[0] /= (self.runs + 1)
            self.summary[1] /= (self.runs + 1)
            self.summary[2] /= (self.runs + 1)
