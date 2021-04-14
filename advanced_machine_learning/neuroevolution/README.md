CS581 Neuroevolution Project
---

**Authors:** Elliot Greenlee and Jared M. Smith

## Overview

Code and assets from our presentation on Neuroevolution.

## Experiments

### XOR

- Runs Done: 5
- Average Generations until solution: 102.4
- Generation count over 5: 205, 78, 59, 109, 61
- Figures in `results/xor` folder.

### Pole Balancing

- Runs Done: 5
- Average Generations until solution: 28
- Generation count over 5: 10, 19, 32, 47, 32
- Figures in `results/single-pole-balancing` folder.

### OpenAI Lunar Lander

- The included OpenAI lunar lander code may be broken in the NEAT-Python project.
- We ran over 12000 generations and never got an average fitness above -0.298.
- Graphs of the fitness and scores over time are in the `results/lunar_lander` folder along with videos from every 10 generations in a run with less than 12000 generations (but still converging to an average fitness of around -.3).
