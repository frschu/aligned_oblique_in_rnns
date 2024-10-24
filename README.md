# aligned_oblique_in_rnns
Repository for code of paper "Aligned and oblique dynamics in recurrent neural networks", Schuessler et al., 2023.
The code should enable to reproduce all results. 


All analysis and plotting was done in notebooks.
Similar to the structure of the paper, the notebooks are structured in 4 blocks:
- `cycling...`: These notebooks generate the example plots based on the cycling task.
- `neuro...`: Simulating and analysing results for different neuroscience tasks.
- `experimental_data...`: Analysis of the experimental data.
- `sine_example...`: The simple sine task that is analyzed in the Methods section.
- `mft_GD...`: Mean field theory for gradient descent analyzed in the Methods section.

Notebooks generating the bulk of the simulations end on `..._train_networks.ipynb`. 
For the different neuroscience tasks, the notebook `neuro_compute_results.ipynb` computes the results for figures comparing across tasks. 
This should be run before any of the `neuro_plot_....ipynb` notebooks, which generate the plots. 

For the results on the experimental data:
- Run the notebook `experimental_data_download.ipynb` first. This will download the datasets (NLB, BCI, Russo).
- Then run the notebook `experimental_data_compute_results.ipynb`. This will compute the statistics to be plotted. Can take some minutes.
- Finally run `experimental_data_plot.ipynb` to generate plots. 

The notebooks `cycling_noise_compression.ipynb`, `linear_rnn_noisy.ipynb` and `mft_mechanism.ipynb` contain both training and plotting. For each, training is relatively short (max 1h).

### Requirements
All code was run with python 3.10.6, with packages specified in `requirements.txt` (to be used within a python venv).
Simulations are run with pytorch. They should run both on CPU and GPU (most recent tests only on CPU).


@ Friedrich Schuessler, 2023
