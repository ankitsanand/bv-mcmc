# Block-Value Symmetries in Probabilistic Graphical Models
This repository contains the code for [Block-Value Symmetries in Probabilistic Graphical Models](https://arxiv.org/abs/1807.00643) - Gagan Madan, Ankit Anand, Mausam, Parag Singla, in Proceedings of Uncertainty in Artificial Intelligence (UAI), Monterey, CA, USA, August 2018.
## Directory Structure
The code is organized as follows:
```
bv_mcmc
├── src: Contains the source code for heuristic, block_gibbs, domains
│   ├── candidate_lists
│   │    ── colour_signaure.cpp: Code for generating block partitions
│   │    ── Makefile: Makefile with rules for colour_signature
│   ├── domain_generation
│   │    ── generate_job_search.py: Generates a PGM from the Job Search Domain
│   │    ── generate_student_curriculum.py: Generates a PGM from the Student Curriculum Domain
│   │    ── add_evidence_job_search.py: Adds evidence to a genreated PGM from Job Search Domain
│   │    ── add_evidence_student_curriculum.py: Adds evidence to a genreated PGM from Job Search Domain
│   │    ── construct_graph.py: Generates an input graph for Saucy
│   │    ── simplify_graph.py: Simplifies a PGM 
│   ├── mcmc
│   │    ── marginals_generator.g: Computes the true marginals by Gibbs sampling for a sufficient number of iterations
│   │    ── gibbs.g: Computes marginals by Vanilla-MCMC and calculates KL-Divergence values with true marginals
│   │    ── nc_gibbs.g: Computes marginals by VV-MCMC and calculates KL-Divergence values with true marginals
│   │    ──  block_mcmc.g: Computes marginals by BV-MCMC and calculates KL-Divergence values with true marginals
├── bin: Contains compiled code for heuristic
│    ──  colour_signature
├── experiments: Contains data for experiments performed
│   ├──  job_search
│   │   ├── 30: Files for 30 people, 0% evidence
│   │   │   ── kl_gibbs.csv: KL-Divergence with true marginals for Vanilla-MCMC
│   │   │   ── nc_kl.csv: KL-Divergence with true marginals for VV-MCMC
│   │   │   ── block_gibbs.csv: KL-Divergence with true marginals for BV-MCMC
│   │   │   ── ...
│   │   ├── 50: Files for 50 people, 0% evidence
│   │   │   ── kl_gibbs.csv: KL-Divergence with true marginals for Vanilla-MCMC
│   │   │   ── nc_kl.csv: KL-Divergence with true marginals for VV-MCMC
│   │   │   ── block_gibbs.csv: KL-Divergence with true marginals for BV-MCMC
│   │   │   ── ...
│   │   ├── 50_10evid: Files for 50 people, 10% evidence
│   │   │   ── kl_gibbs.csv: KL-Divergence with true marginals for Vanilla-MCMC
│   │   │   ── nc_kl.csv: KL-Divergence with true marginals for VV-MCMC
│   │   │   ── block_gibbs.csv: KL-Divergence with true marginals for BV-MCMC
│   │   │   ── ...
│   ├──  student_curriculum
│   │   ├── 600_50: Files for 600 students, 50 friends, 0% evidence
│   │   │   ── kl_gibbs.csv: KL-Divergence with true marginals for Vanilla-MCMC
│   │   │   ── nc_kl.csv: KL-Divergence with true marginals for VV-MCMC
│   │   │   ── block_gibbs.csv: KL-Divergence with true marginals for BV-MCMC
│   │   │   ── ...
│   │   ├── 1200_100: Files for 1200 students, 100 friends 0% evidence
│   │   │   ── kl_gibbs.csv: KL-Divergence with true marginals for Vanilla-MCMC
│   │   │   ── nc_kl.csv: KL-Divergence with true marginals for VV-MCMC
│   │   │   ── block_gibbs.csv: KL-Divergence with true marginals for BV-MCMC
│   │   │   ── ...
│   │   ├── 1200_100_10evid: Files for 1200 students, 100 friends 10% evidence
│   │   │   ── kl_gibbs.csv: KL-Divergence with true marginals for Vanilla-MCMC
│   │   │   ── nc_kl.csv: KL-Divergence with true marginals for VV-MCMC
│   │   │   ── block_gibbs.csv: KL-Divergence with true marginals for BV-MCMC
│   │   │   ── ...
```
## Compilation Instructions
Compilating the code for heuristic requires access to saucy source code. If you have access to saucy source code, please paste the files src/candidate_lists/colour_signature.cpp and src/candidate_lists/Makefile to your saucy directory and then compile using the Makefile:
```
cp src/candidate_lists/* <path-for-saucy-directory>
make colour_signature
```
This should generate colour_signature binary. Run this using:
```
./colour_signature <number_of_candidate_lists_to_generate>
```
In case the number of candidate lists to generate is not specified, this would try to use as many candidate lists required to cover all valid blocks.

## Generating Domains and Running MCMC
To generate instances from a new domain and run MCMC, follow these steps:
1. Generate the domain PGM using:
```
cd src/domain_generation
python generate_job_search.py <num_people> test_raw
python generate_student_curriculum.py <num_students> <num_friends> test_raw
```
2. (Optional) Add evidence to the generated PGM
```
python add_evidence_job_search.py <evidence_fraction> test_raw test
python add_evidence_student_curriculum.py <evidence_fraction> test_raw test
```
3. Simplify graph and construct graph for input to Saucy
```
python simplify_graph.py test test.num
python construct_graph.py test test.saucy
```
4. Generate true marginals for the given PGM
```
$GAP_DIR/bin/gap.sh -q < marginals_generator.g
```
5. Run Vanilla-MCMC, VV-MCMC and BV-MCMC
```
$GAP_DIR/bin/gap.sh -q < gibbs.g
$GAP_DIR/bin/gap.sh -q < nc_gibbs.g
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
```
## Citing this work
```
Block-Value Symmetries in Probabilistic Graphical Models - Gagan Madan, Ankit Anand, Mausam, Parag Singla. In Proceedings of Uncertainty in Artificial Intelligence (UAI), Monterey, CA, USA, August 2018.
```
