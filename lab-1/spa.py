import pprint
import random

from typing import List


class Genome:
    _genome: List[int]

    def __init__(self, size: int):
        self._genome = [random.randint(0, 1) for _ in range(size)]

    def mutation(self, mutation_rate: float, mutation_prob: float) -> 'Genome':
        if not (1 >= mutation_rate >= 0 and 1 >= mutation_prob >= 0):
            raise Exception('mutation rate and prob must be between 0 and 1')
        if mutation_prob < random.random():
            genome_string = [1 - bit if random.random() < mutation_rate else bit for bit in self._genome]

            genome = Genome(len(self._genome))
            genome._genome = genome_string
            return genome
        else:
            return self

    def fitness(self) -> int:
        return sum(self._genome)

    def crossover(self, partner: 'Genome', cross_prob: float) -> ('Genome', 'Genome'):
        if random.random() < cross_prob:
            cross_over_point = random.randint(0, len(self._genome) - 1)
            g1 = Genome(len(self._genome))
            g2 = Genome(len(self._genome))
            g1._genome = self._genome[:cross_over_point] + partner._genome[cross_over_point:]
            g1._genome = partner._genome[:cross_over_point] + self._genome[cross_over_point:]
            return g1, g2
        else:
            return self, partner

    def __repr__(self):
        return str(self._genome)


def random_pop(pop_size: int, genome_size: int):
    return [Genome(genome_size) for _ in range(pop_size)]


def select(pop: List[Genome], fits: List[int], pop_size: int):
    return random.choices(pop, fits, k=pop_size)


def crossover(pop: List[Genome], crossover_prob: float):
    off = []
    for genome1, genome2 in zip(pop[::2], pop[1::2]):
        off.extend(genome1.crossover(genome2, crossover_prob))
    return off


def mutation(pop: List[Genome], mutation_rate: float, mutation_prob: float):
    return [genome.mutation(mutation_rate, mutation_prob) for genome in pop]


def main():
    POP_SIZE = 100
    MAX_GEN = 100
    GENOME_SIZE = 25
    CROSS_PROB = 0.8
    MUT_PROB = 0.2
    MUT_RATE = 1 / GENOME_SIZE
    pop: List[Genome] = random_pop(POP_SIZE, GENOME_SIZE)
    for i in range(MAX_GEN):
        fits = [genome.fitness() for genome in pop]
        print(f'gen {i}: {max(fits)}')
        mating_pool = select(pop, fits, POP_SIZE)
        off = crossover(mating_pool, CROSS_PROB)
        off = mutation(off, MUT_RATE, MUT_PROB)
        pop = off[:]

    pprint.pprint(pop)


if __name__ == '__main__':
    main()
