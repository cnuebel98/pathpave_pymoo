This is the README 


Parameters for Experiments:
31 seeds per run
4 maps (gradient, radial-gradient, river, sinusoidal)
2 Crossover (OnePoint, TwoPoint)
6 weight Shifting Strategies (Random, leastResistance, MaxResistance, SplitInTwo, SplitInThree, EqualDistribution)
10 algorithms (NSGA2, NSGA3, SPEA2, MOEAD, RNSGA2, AGEMOEA, AGEMOEA2, DNSGA2, SMSEMOA, CTAEA)

Constant:
everything else
pop_size = 100
n_evals = 100.000
map size: 51,51
start, end: middle bottom, middle top
ref_dirs = das_dennis mit 10 partitions
repair: Pathrepair
duplicateElimination
callback
mutation: RadiusSamplingMutation
Sampling: RandomSampling

