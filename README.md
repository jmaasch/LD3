# Local Causal Discovery for Structural Evidence of Direct Discrimination

This repository contains source code and simulation notebooks for *local discovery for direct discrimination* (LD3). LD3 is a local causal discovery method that is designed to assess the structural direct criterion and facilitate weighted controlled direct effect estimation.

## Repository structure
```bash
.
├── README.md
├── cde_simulation.ipynb: weighted cde estimation on linear-gaussian dags
├── data
│   ├── erdos_renyi: multiple .csv files
│   └── star_liver_data
│       ├── star_liver_data_dictionary.xlsx: describes feature levels for categorical variables
│       ├── summary_stats_17_19.csv
│       ├── summary_stats_20_22.csv
│       ├── summary_stats_significant_17_19.csv
│       └── summary_stats_significant_20_22.csv
├── data_generation.py: data generation for a linear-gaussian dag
├── environment.yml
├── ld3.py: source code for main method
├── requirements.txt
└── simulations_oracle_erdos.ipynb: experiments with erdos-renyi graphs
```
