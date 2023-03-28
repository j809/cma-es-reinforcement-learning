import csv
import numpy as np
from scipy import stats
from multiprocessing import Pool
import multiprocessing
from functools import partial
from tqdm import tqdm
import random
import cma


class TimeStep:
    state = 0
    action = 0
    reward = 0
    pi = 0


class Episode:
    time_steps = []


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


# Modified this wrapper to accept random seed (this class turns CMA minimizer to maximizer)
class CMAES:
    def __init__(self,
                 num_params,
                 sigma_init=0.10,
                 popsize=255,
                 weight_decay=0.01,
                 seed=0):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None

        self.es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                           self.sigma_init,
                                           {'popsize': self.popsize,
                                            'seed': seed})

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma*sigma))

    def ask(self):
        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        self.es.tell(self.solutions, reward_table.tolist())

    def current_param(self):
        return self.es.result[5]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]

    def result(self):
        r = self.es.result
        return r[0], -r[1], -r[1], r[6]


def parse_file(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        is_first_line = True
        all_episodes = []
        num_episodes = 0
        time_step = None
        time_steps = 0
        cur_episode = None
        print_line = False
        episode_count = 0

        for row in csv_reader:
            if is_first_line:
                num_episodes = int(row[0])
                is_first_line = False
            elif time_steps == 0:
                time_steps = int(row[0])
                if cur_episode is not None:
                    all_episodes.append(cur_episode)

                cur_episode = Episode()
                cur_episode.time_steps = []
                episode_count += 1
                print_line = True
            else:
                time_steps -= 1
                time_step = TimeStep()
                time_step.state = int(row[0])
                time_step.action = int(row[1])
                time_step.reward = int(row[2])
                time_step.pi = float(row[3])
                cur_episode.time_steps.append(time_step)

            if print_line and episode_count % 100000 == 0:
                print(f'processed {episode_count} episodes')
                print_line = False

        return all_episodes


def stddev(lst):
    return np.std(lst, ddof=1)


def get_action_probabilities(theta):
    action_probabilities = np.exp(theta)
    action_probabilities = np.divide(action_probabilities, np.sum(action_probabilities))
    return action_probabilities


def get_pdis_for_episode(theta, episode):
    pdis = 0
    gamma = 1
    ratio = 1
    step = 0

    if theta is not None:
        theta = theta.reshape((NUM_STATES, NUM_ACTIONS))
        # pi = compute_pi(theta)

    for time_step in episode.time_steps:
        if theta is None:
            pi_e_action = time_step.pi
        else:
            pi_e_action = get_action_probabilities(theta[time_step.state, :])[time_step.action]
        pi_b_action = time_step.pi

        ratio = ratio * (pi_e_action / pi_b_action)
        pdis = pdis + gamma * ratio * time_step.reward
        gamma *= GAMMA
        step += 1

    return pdis


def get_pdis_for_all_episodes(data, theta=None):
    pdis_list = []

    for episode in data:
        pdis_list.append(get_pdis_for_episode(theta, episode))

    pdis_mean = np.average(pdis_list)
    pdis_stddev = stddev(pdis_list)
    return pdis_mean, pdis_stddev


def candidate_selection(candidate_data, safety_data_size, j_behavior, theta):

    pdis_c, pdis_c_stddev = get_pdis_for_all_episodes(candidate_data, theta)

    tinv = stats.t.ppf(1 - DELTA, safety_data_size - 1)

    check = pdis_c - 2 * pdis_c_stddev / np.sqrt(safety_data_size) * tinv - j_behavior

    if check >= 0:
        return pdis_c
    else:
        return pdis_c - 1000  # barrier penalty


def safety_test(safety_data, safety_data_size, j_behavior, theta):

    pdis_s, pdis_s_stddev = get_pdis_for_all_episodes(safety_data, theta)

    tinv = stats.t.ppf(1 - DELTA, safety_data_size - 1)

    check = pdis_s - pdis_s_stddev / np.sqrt(safety_data_size) * tinv - j_behavior

    if check >= 0:
        return True
    else:
        return False


def optimizer(solver, candidate_data, safety_data_size, j_behavior, num_iterations):
    history = []
    for j in range(num_iterations):
        solutions = solver.ask()

        func = partial(candidate_selection, candidate_data, safety_data_size, j_behavior)
        fitness_list = []

        num_cores = multiprocessing.cpu_count()
        pool = Pool(processes=num_cores)

        for x in tqdm(pool.imap_unordered(func, solutions), total=len(solutions)):
            fitness_list.append(x)

        pool.close()
        pool.join()

        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j + 1) % 1 == 0:
            print("fitness at iteration", (j + 1), result[1])

        # print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history, result[0]


def simulate(theta, transition_prob, reward_function):
    theta = theta.reshape((18, 4))
    pi_e = np.zeros((18, 4))

    for i in range(18):
        prob = np.exp(theta[i, :])
        prob /= np.sum(prob)
        pi_e[i, :] = prob

    print("trying estimate policy")
    ge_total = 0
    for x in range(5000):
        s = 17
        gamma = 0.95
        g_e = 0
        step = 0
        while True:
            if len(pi_e[s, :]) == 0 or step == 200:
                break
            a = np.random.choice(4, 1, p=pi_e[s, :])
            r = np.random.choice(list(reward_function[(s, a[0])].keys()), 1, p=list(reward_function[(s, a[0])].values()))
            g_e += (gamma ** step) * r[0]
            step += 1
            # print(f"step {step} done with State {s}, Action {a[0]} and Reward {r[0]} with Return {g_e}")
            if (s, a[0]) not in transition_prob:
                break
            s = list(transition_prob[(s, a[0])].keys())
            s = int(s[0])
        # print(f'{x} episode return {g_e}')
        ge_total += g_e

    return ge_total/5000


def generate_global_functions(data):
    pi = np.zeros((18, 4))
    transition_prob = {}
    reward_function = {}

    for episode in data:
        first = True
        p = (episode.time_steps[0].state, episode.time_steps[0].action)

        for time_step in episode.time_steps:
            c = (time_step.state, time_step.action)
            pi[time_step.state, time_step.action] = time_step.pi

            if c not in reward_function:
                reward_function[c] = {time_step.reward: 1}
            elif time_step.reward not in reward_function[c]:
                reward_function[c][time_step.reward] = 1
            else:
                reward_function[c][time_step.reward] += 1

            if first:
                first = False
                continue
            if p not in transition_prob:
                transition_prob[p] = {c[0]: 1}
            elif c[0] not in transition_prob[p]:
                transition_prob[p][c[0]] = 1
            else:
                transition_prob[p][c[0]] += 1

            p = c

    for key, value in transition_prob.items():
        s = 0
        for k, v in value.items():
            s += v
        for k, v in value.items():
            v /= s
            transition_prob[key][k] = v

    for key, value in reward_function.items():
        s = 0
        for k, v in value.items():
            s += v
        for k, v in value.items():
            v /= s
            reward_function[key][k] = v

    rewards = [0, 1, 10]
    for key, value in reward_function.items():
        for r in rewards:
            if r not in reward_function[key]:
                reward_function[key][r] = 0

    print("trying behavior policy")
    gb_total = 0
    for x in range(5000):
        s = 17
        gamma = 0.95
        g_b = 0
        step = 0
        while True:
            if len(pi[s, :]) == 0 or step == 200:
                break
            a = np.random.choice(4, 1, p=pi[s, :])
            r = np.random.choice(list(reward_function[(s, a[0])].keys()), 1, p=list(reward_function[(s, a[0])].values()))
            g_b += (gamma ** step) * r[0]
            step += 1
            # print(f"step {step} done with State {s}, Action {a[0]} and Reward {r[0]} with Return {g_b}")
            if (s, a[0]) not in transition_prob:
                break
            s = list(transition_prob[(s, a[0])].keys())
            s = int(s[0])
        # print(f'{x} episode return {g_b}')
        gb_total += g_b

    return transition_prob, reward_function, gb_total/5000


NUM_ACTIONS = 4
NUM_STATES = 18
DELTA = 0.01
GAMMA = 0.95
NUM_ITERATIONS = 8
NUM_PARAMS = 72
NUM_POPULATION = 16

SEED = 0


if __name__ == '__main__':
    all_episodes_data = parse_file('data.csv')

    print("calculating global functions")
    transition_p, r_function, gb_t = generate_global_functions(all_episodes_data)
    print(f"simulated behavior return {gb_t}")

    total_len = len(all_episodes_data)

    # please uncomment the lines below to reproduce high return policies

    # all_episodes_data = all_episodes_data[:25000]
    # split = 20000
    # candidate_data_end_index = int(split)
    # data_candidate = all_episodes_data[:candidate_data_end_index]
    # data_safety = all_episodes_data[candidate_data_end_index:]
    # data_safety_size = len(data_safety)

    candidate_data_end_index = int(0.8 * total_len)  # run on 80% candidate data, 20% safety data
    data_candidate = all_episodes_data[: candidate_data_end_index]
    data_safety = all_episodes_data[candidate_data_end_index]
    data_safety_size = len(data_safety)

    j_b = 1.4137  # kept constant as mentioned in given problem

    policies_accepted = 0

    while True:
        if policies_accepted == 100:
            break

        SEED += 1
        random.seed(SEED)
        np.random.seed(SEED)

        print(f'setting random seed as {SEED}')

        # please uncomment the lines below to reproduce all the policies submitted (mix of high return policies and
        # policies generated from smaller data set since I could not run the generation of high return policies for
        # longer due to limited time and resources)

        # if SEED <= 80:
        #     all_episodes_data = all_episodes_data[:25000]
        #     split = 20000
        #     candidate_data_end_index = int(split)
        #     data_candidate = all_episodes_data[:candidate_data_end_index]
        #     data_safety = all_episodes_data[candidate_data_end_index:]
        #     data_safety_size = len(data_safety)
        # else:
        #     all_episodes_data = all_episodes_data[:12000]
        #     split = 2000
        #     candidate_data_end_index = int(split)
        #     data_candidate = all_episodes_data[:candidate_data_end_index]
        #     data_safety = all_episodes_data[candidate_data_end_index:]
        #     data_safety_size = len(data_safety)

        cmaes = CMAES(NUM_PARAMS,
                      popsize=NUM_POPULATION,
                      weight_decay=0.0,
                      sigma_init=0.3,
                      seed=SEED)

        cma_history = optimizer(cmaes, data_candidate, data_safety_size, j_b, NUM_ITERATIONS)

        theta_e = cma_history[1]
        all_fitness = cma_history[0]

        if all_fitness[-1] <= j_b:
            continue

        safety_check = safety_test(data_safety, data_safety_size, j_b, theta_e)

        print(f"Safety check result {safety_check}")
        print(theta_e)

        ge_t = simulate(theta_e, transition_p, r_function)
        print(f"Theta_e simulation return {ge_t}")

        simulate_check = False
        if ge_t - j_b >= 0.8:
            simulate_check = True

        if safety_check and simulate_check:
            policies_accepted += 1
            with open("Policies/policy" + str(policies_accepted) + ".txt", 'w') as f:
                for i in range(len(theta_e)):
                    print(theta_e[i])
                    f.write(str(theta_e[i]) + '\n')
            with open("Seeds/seed" + str(policies_accepted) + ".txt", 'w') as f:
                f.write(str(SEED) + '\n')
        elif safety_check:
            policies_accepted += 1
            with open("Policies/policy" + str(policies_accepted) + ".txt", 'w') as f:
                for i in range(len(theta_e)):
                    print(theta_e[i])
                    f.write(str(theta_e[i]) + '\n')
            with open("Errors/error" + str(policies_accepted) + ".txt", "w") as f:
                f.write("Theta E failed simulation check with return" + str(ge_t) + "\n")
            with open("Seeds/seed" + str(policies_accepted) + ".txt", 'w') as f:
                f.write(str(SEED) + '\n')
