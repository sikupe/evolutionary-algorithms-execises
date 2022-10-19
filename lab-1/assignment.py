import random
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def fitness(ind):
    max_length = 0
    i = 0
    while i < len(ind):
        previous = None
        length = 0
        for j in range(i, len(ind)):
            if ind[j] != previous:
                previous = ind[j]
                length += 1
            else:
                i += length - 1
                break
        if max_length < length:
            max_length = length
        i += 1

    return max_length


def random_ind(length):
    return [random.randint(0, 1) for _ in range(length)]


def random_init(size, ind_len):
    return [random_ind(ind_len) for _ in range(size)]


def select(pop, fits, N):
    return random.choices(pop, fits, k=N)


def cross(p1, p2):
    cp = random.randrange(1, len(p1))
    o1 = p1[:cp] + p2[cp:]
    o2 = p2[:cp] + p1[cp:]
    return o1, o2


def crossover(pop, cx_prob):
    off = []

    for p1, p2 in zip(pop[::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)

    return off


def mutate(p, mut_prob_per_bit):
    o = []
    for g in p:
        if random.random() < mut_prob_per_bit:
            g = 1 - g
        o.append(g)
    return o


def mutation(pop, mut_prob, mut_prob_per_bit):
    off = []

    for p in pop:
        if random.random() < mut_prob:
            o = mutate(p, mut_prob_per_bit)
        else:
            o = p[:]
        off.append(o)

    return off


def evolutionary_algorithm(pop_size, ind_len, max_gen, cross_prob, mut_prob, mut_prob_per_bit):
    log = []
    avg = []
    pop = random_init(pop_size, ind_len)
    for G in range(max_gen):
        fits = [fitness(ind) for ind in pop]
        log.append(max(fits))
        avg.append(np.average(fits))
        mating_pool = select(pop, fits, pop_size)
        off = crossover(mating_pool, cross_prob)
        off = mutation(off, mut_prob, mut_prob_per_bit)
        pop = off[:]

    return pop, log, avg


def main():
    if not os.path.exists('results'):
        os.mkdir('results')

    IND_LEN = 25
    POP_SIZE = 100
    MAX_GEN = 500
    CROSS_PROB = 0.8
    MUT_PROB = 0.1
    MUT_PROB_PER_BIT = 1 / IND_LEN

    for cross_prob in [0, .2, .4, .6, .8, 1]:
        for mut_prob in [.1, .2, .5, .8]:
            print(f'Running with cross prob {cross_prob} and mutation prob {mut_prob}')

            rounds = 100
            pops = np.empty((POP_SIZE, IND_LEN, rounds))
            logs = np.empty((MAX_GEN, rounds))
            avgs = np.empty((MAX_GEN, rounds))
            for i in tqdm(range(rounds)):
                pop, log, avg = evolutionary_algorithm(POP_SIZE, IND_LEN, MAX_GEN, cross_prob, mut_prob,
                                                       MUT_PROB_PER_BIT)
                pops[:, :, i] = np.array(pop)
                logs[:, i] = np.array(log)
                avgs[:, i] = np.array(avg)

            # fits = np.apply_along_axis(fitness, axis=1)
            # fits = np.squeeze(fits, axis=1)
            # pprint.pprint(list(zip(fits, pop)))
            # print(f'Max fitness: {max(fits)}')
            plt.title(f"SGA - Crossover prob: {cross_prob} - Mutation prob: {mut_prob}")
            plt.plot(np.mean(logs, axis=1), label="Max")
            plt.plot(np.mean(avgs, axis=1), label="Avg")
            plt.legend(loc='best')
            plt.savefig(f'results/plt-{cross_prob}-{mut_prob}.png')
            plt.clf()
            plt.close()


if __name__ == '__main__':
    main()
