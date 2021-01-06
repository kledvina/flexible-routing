# flexible-routing
The flexible-routing repository contains simulation code and output files to computationally assess the costs of a vehicle routing strategy with fixed routes and customer sharing. This work improves on the simulation study in the working paper by [Ledvina et al. (2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3656374) and serves as supplemental files for Kirby Ledvina's masters thesis in MIT's Department of Civil and Environmental Engineering.

## Overview

We explore vehicle routing with customer sharing as a strategy to accomodate new and variable customer demands in distribution networks with fixed delivery routes. For this setting, we propose predesigning routes with some overlap such that adjacent routes share customers. This design gives the delivery fleet operator the flexibility to assign drivers to subsets of their predesigned routes in response to realized customer demands.

We define four alternative strategies for designing and executing routes with varying degrees of flexibility: dedicated routing, overlapped routing, full flexibility, and reoptimization. The Jupyter notebook *routing_examples.ipynb* describes and illustrates these  routing strategies for a randomly generated example problem. Please see [Ledvina et al. (2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3656374) for a formal description of our routing model.

## Contents

This repository includes simulation code (Python), output data and figures, and an R script for creating summary figures from the data.

**Simulation Code**
- *simulate.py* - defines new scenarios and runs the simulation
- *supporting.py* - includes supporting code such as functions to execute routing algorithms and calculate transportation costs
- combine_outputs.py 

**Simulation Output**
- *output/* subfolder
- *combined_outputs_2020-10-23.xlsx*

**Graphing Files**
- *create_figures.R*
- *figures/* subfolder

Finally, the Jupyter notebook *routing_examples.ipynb* is a supplemental file that allows users to generate a random customer and demand instance and see the resulting routes and costs under the different routing strategies.

## Running Simulations
TODO

## Contributors

**Code Author:** Kirby Ledvina, MIT Data Science Lab, kirby.ledvina@gmail.com

**Collaborators**
- Hanzhang Qin, MIT Data Science Lab, hqin@mit.edu
- Prof. David Simchi-Levi (Advisor), MIT Data Science Lab, dslevi@mit.edu
- Prof. Yehua Wei, Duke Fuqua School of Business, yehua.wei@duke.edu


