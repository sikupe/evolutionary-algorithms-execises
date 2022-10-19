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
I took the code which we developed in the practical and adapted it to the assignment tasks:
We shall submit a pot comparing two different settings of the algorithm (alternating genome).
In order to make the SGA suitable to optimize the genome to have only alternating series of 1s and 0s, the fitness function must be adapted.
I did that using the maximum length of an alternating series within the genome as fitness value. This is done by having two nested loops, the first iterating over all possible start indices of a series and the second loop counting the alternating 0s and 1s from each start position.
I ran the algorithm 100 rounds for each setting and took the mean of the fitness values over those runs per generation. Also I took the mean over the fitness values of the best individual of each generation over the rounds.
Two example settings can be found attached, where the mean of the best individuals over the rounds and the average population fitness is plotted per generation.

