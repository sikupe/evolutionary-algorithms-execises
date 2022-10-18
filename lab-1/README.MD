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


Selection
$p_i =\frac{f_i, \sum_{i} f_i}$
1. randomly select two individuals i_1, i_2
2. if fit(i_1) > fit(i_2)
   1. select i_1 with high probs (80%)
3. else
   1. select i_2
