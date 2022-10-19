# Problem: OneMAX
Maximize the number of binary ones in individual

$$F(x) = \sum_{i}=x_i$$

# Steps
1. Create an initial (random) population
2. Fits <- fitness of pop
3. while not happy
   1. mating-pool <- selection (pop, fits)
   2. off <- crossover
   3. off <- mutation
   4. pop <- off


# Selection
$p_i =\frac{f_i, \sum_{i} f_i}$
1. randomly select two individuals i_1, i_2
2. if fit(i_1) > fit(i_2)
   1. select i_1 with high probs (80%)
3. else
   1. select i_2


# Description of what I did
1. Implement the simple genetic algorithm in a programming language of your choice.
Implemented as described in the tutorial: Used a list of integers in order to represent the binary string for the genome

2. Use the implemented genetic algorithm to find an individual with all 1s. (So called OneMAX problem.)
Used as a fitness function the sum over the genome (as it can only contain 0s and 1s and we want to maximize the 1s)
For the recombination an 1-point-crossover was used with a probability of .8 and for the mutation bit inversion was used with a probability of .2 for the mutation of an individual and 1 / (length of individual) for the mutation of each bit.
The individuals had a genome string of length 25, the population was 100 and I generated 100 generations.
In order to get a more comparable result the algorithm was run over 100 rounds and the mean over those rounds was taken per generation.
-> As shown in the tutorial

3. Change the fitness of the algorithm to find an individual with alternating 1s and 0s (either 010101... or 101010...).
In order to make the SGA suitable to optimize the genome to have only alternating series of 1s and 0s, the fitness function must be adapted. I did that using the maximum length of an alternating series within the genome as fitness value. This is done by having two nested loops, the first iterating over all possible start indices within the genome string and the second loop counting length the series of alternating 0s and 1s from each start position.
I ran the algorithm 100 rounds for each setting and took the mean of the fitness values over those runs per generation. Also I took the mean over the fitness values of the best individual of each generation over the rounds.

4. Try to change the parameters (probability of mutation/crossover) and see what happens.
Depending on the configuration the SGA converges sometimes faster and sometimes slower, also there is a big variance between the configuration regarding the difference of the best individual and the mean individual.
In general a higher crossover rates led to faster convergence, but didn't affect the distance between the mean individual and the best individual, where a high mutation rate led to a faster convergence but also a very high distance between the best individual and the mean individual of the final population.

Tested configurations: crossover prob ∈ {0, .2, .4, .6, .8, 1}, mutation prob ∈ {.1, .2, .5, .8}.

The best result regarding convergence, average fitness in the final population and fitness of the best individual had crossover prob = 1 and mutation prob = 0.1

5. Submit a plot comparing the convergence of the algorithm for two different settings of the algorithm.
Two example settings can be found attached, where the mean of the best individuals over the rounds and the average population fitness is plotted per generation.
Therefore the plotting is done differently than in the second tutorial proposed, I still find it useful, as it shows the difference in how good the algorithm creates "elite" individuals compared to the average fitness of the population

The plots show the best configuration and the worst configuration in my case:
Best: crossover rate 1, mutation rate .1
Worst: crossover rate 0, mutation rate .8

6. Explain what you did.
See above.








