import random
import pprint
import matplotlib.pyplot as plt
import numpy as np

IND_LEN = 25
POP_SIZE = 100
MAX_GEN = 100
CROSS_PROB = 0.8
MUT_PROB = 0.2
MUT_PROB_PER_BIT = 1 / IND_LEN


def fitness(ind):
    return sum(ind)


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


def evolutionary_algorithm():
    log = []
    avg = []
    pop = random_init(POP_SIZE, IND_LEN)
    for G in range(MAX_GEN):
        fits = [fitness(ind) for ind in pop]
        log.append(max(fits))
        avg.append(np.average(fits))
        mating_pool = select(pop, fits, POP_SIZE)
        off = crossover(mating_pool, CROSS_PROB)
        off = mutation(off, MUT_PROB, MUT_PROB_PER_BIT)
        pop = off[:]

    return pop, log, avg


def main():
    pop, log, avg = evolutionary_algorithm()
    fits = [fitness(ind) for ind in pop]
    pprint.pprint(list(zip(fits, pop)))
    print(f'Max fitness: {max(fits)}')
    plt.plot(log, label="Max")
    plt.plot(avg, label="Avg")
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
