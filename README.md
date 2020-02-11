# SAPMLAB
EEG Analysis for the signal processing and machine learning analysis for biomedical data course
Topic: Reversal Learning and Morbus Parkinson
## Contact
Julius Kricheldorff M.Sc. - julius.kricheldorff@uni-oldenburg.de  
Department of Neurology, Carl von Ossietzky University Oldenburg
## Content
This repository contains all necessary scripts and links to perform a SVM analysis on Go/NoGo EEG task data.
The purpose of this project was the analysis of a Go/NoGo EEG data set via machine learning techniques. The data set used for this analysis is a subset (five participants) of the publicly avaialable [data set] by Swart et al. 2018. For the analysis preprocessing was performed with help of the pre-processing scripts (minorly adapted) kindly provided by Swart et al. 2018. Pre-processing was performed using the [Fieldtrip toolbox] (Oostenveld, Fries, Maris and Schoffelen 2011). After scaling and down-sampling via a sliding window approach (40ms bins with 20ms overlap), data were exported to Python for the ML_analysis. In Python using [scikit-learn] I performed a logistic ridge regression analysis to test for linear patterns in the data and a SVM analysis using a radial-basis function kernel to test for non-linear patterns.

## Bibliography
1. Oostenveld, R., Fries, P., Maris, E., Schoffelen, JM (2011). FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data. Computational Intelligence and Neuroscience, Volume 2011 (2011), Article ID 156869,
https://doi.org/10.1155/2011/156869

2. Swart JC, Frank MJ, Määttä JI, Jensen O, Cools R, et al. (2018) Frontal network dynamics reflect neurocomputational mechanisms for reducing maladaptive biases in motivated action. PLOS Biology 16(10): e2005979. https://doi.org/10.1371/journal.pbio.2005979

[data set]: https://public.data.donders.ru.nl/dccn/DSC_3017033.03_624:v1/
[Fieldtrip toolbox]: http://www.fieldtriptoolbox.org
[scikit-learn]: https://scikit-learn.org/stable/index.html
