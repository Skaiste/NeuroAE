# Predicting Alzheimer's evolution using Autoencoders
This project focuses on artificial intelligence and deep learning techniques to simulate the progression of Alzheimer's disease. In particular, neural networks based on autoencoders and their variants are proposed as tools to predict and visualize the evolution of the disease throughout its clinical stages. The research builds on knowledge of neural networks acquired through specialized courses and a review of the scientific literature. Real patient data from the ADNI database, covering all clinical phases (HC, MCI, and AD), will be used. Throughout the project, models will be developed using Deep Learning tools such as autoencoders, variational autoencoders (VAE), and convolutional autoencoders (CAE). Each model will be employed both to reconstruct the input signals and to carry out the main experiment: generating neuronal deep fakes by combining a general encoder trained on all clinical groups with the decoder of a specific group. During inference applying the combined model of a different clinical group, aiming to obtain reliable simulations of disease progression. The innovative approach proposed in this study will open new avenues for future research on diagnostic biomarkers in this field.

## Dataset
For this study, real patient data obtained from the ADNI database, specifically the matching dataset, is used. The subjects were classified into three groups based on disease status:
1. HC (Healthy Controls): Healthy controls (105 subjects).
2. MCI (Mild Cognitive Impairment): Mild cognitive impairment (90 subjects).
3. AD (Alzheimer’s Disease): Alzheimer’s disease (39 subjects).

The data were obtained using fMRI (functional magnetic resonance imaging), a neuroimaging technique that measures brain activity by detecting changes in blood flow associated with neuronal activity. This method allows the study of the functional connectivity of the brain and is fundamental for the analysis of neurodegenerative diseases such as Alzheimer’s. It should be noted that the ADNI data were captured with a time interval of 3 seconds between samples. According to the Nyquist theorem, this implies that the maximum frequency that can be reliably analyzed is 0.18 Hz. In this study, however, the analysis was limited to a frequency band between 0.01 Hz and 0.08 Hz, as this range has been shown to contain relevant information for the study of resting-state brain connectivity. 
For data analysis and processing, a Schaefer 2018 400-region partitioning was applied, a brain segmentation method that divides the cerebral cortex into 400 distinct areas based on functional characteristics. This partitioning facilitates the extraction of brain activity patterns in a structured and coherent manner.

## Setup
Requirements: python 3.9 & [uv](https://docs.astral.sh/uv/guides/install-python/)
To setup project run:
```bash
uv sync
```

To run training make sure the data (`.mat` files) are in the `data` directory, then run command:
```bash
uv run neuroae
```

## Configuration
Full configuration reference for all models and config files:
- `docs/configuration.md`

## Implementation Checklist
- [x] Data Loading
- [x] Training Framework base
- [x] Evaluation Framework base
- [x] Data pre-processing (filtering + normalisation)
- [x] Baseline models
- [ ] Model improvements
- [x] Framework for parameter tuning

#### Todo's:
- keep reading papers and start looking into other types of autoencoders to implement
- explore normalisation region based and not per feature when flattening a sample
- for experimentation purposes only, train a general VAE, run inference on all data, to extract latent properties and try out clustering algorithms to see if any most clear clusters could come close to the labels

## Report Writing Checklist
- [ ] [ 0% ] State of the art
- [ ] [ 0% ] Specification and design of the solution
- [ ] [ 0% ] Development of the proposal/technique/work that has been carried out
- [ ] [ 0% ] Experimentation and evaluation of the proposal/technique/work
- [ ] [ 0% ] Sustainability analysis and ethical implications
- [ ] [ 0% ] Summary, Introduction, motivation and objectives, Conclusion


In the end report should contain:
- Cover page
- Acknowledgements (ADNI & people who parcelated data)
- Summary
- Index
- Introduction, motivation and objectives
- State of the art
- Specification and design of the solution
- Development of the proposal/technique/work that has been carried out
- Experimentation and evaluation of the proposal/technique/work
- Sustainability analysis and ethical implications
- Conclusions
- List of references used
- Annexes with complementary information
