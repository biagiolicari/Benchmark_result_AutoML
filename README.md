# HOW TO INSTALL AND EXECUTE

### Be sure to have installed python, Conda and R in your environment.


## Execute the frameworks

- In a terminal, execute the command
```console
 cd [framework_folder]
 ```

- Inside the framework folder there is a script file called run_[framework name]_experiments.sh, in the terminal run 
```console
chmod +x run_[framework name]_experiments.sh
```

- After that, type
```console
 ./run_[framework name]_experiments.sh
 ``` 
 this script permits to create a conda environment with the required packages and then executes the framework.


# Reproduce paper result
- In the main folder, benchmark_reproducibility.ipynb permits to recreate the same result obtained in the paper making use of the data already collected from the experiments.
- Figures created from this script will be saved in the directory "figures"
- To conduct the BBT analysis, after executing the script benchmarking_reproducibility.py" be sure to execute the "Bayesian_Bradley_Terry_tree.ipynb" tree. This file makes use of R as a programming language