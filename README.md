# aligned_oblique_in_rnns
Repository for code of paper "Aligned and oblique dynamics in recurrent neural networks", Schuessler et al., 2023.
The code should enable to reproduce all results. 

Similar to the structure of the paper, the notebooks are structured in 4 blocks:
- `cycling...`: These notebooks generate the example plots based on the cycling task.
- `neuro...`: Simulating and analysing results for different neuroscience tasks.
- `experimental_data...`: Analysis of the experimental data.
- `sine_example...`: The simple sine task that is analyzed in the Methods section.

For the results on the experimental data:
- Run the notebook `experimental_data_download.ipynb` first. This will download the datasets (NLB, BCI, Russo).
- Then run the notebook `experimental_data_run_analysis.ipynb`. This will compute the statistics to be plotted. Can take some minutes.
- Finally run `experimental_data_plot.ipynb` to generate plots. 


### Requirements
All code was run with python 3.10.6, with packages specified in `requirements.txt`.
All analysis and plotting was done in notebooks. Notebooks generating the bulk of the simulations end on `...gen.ipynb`. 
Simulations are run with pytorch. They should run both on CPU and GPU (most recent tests only on CPU).
For plotting, a recent version of Latex should be installed.

@ Friedrich Schuessler, 2023