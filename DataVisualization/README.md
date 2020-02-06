## Data Visualization Script
Simple script to skim the data, have a look at the ERPs and see what kind of ERP differences are to be expected.

### Notes for the analysis of the behavioral data
The actual task consisted of a motivational Go/NoGo task. Hence there were multiple different types of Go and NoGo conditions. For the analysis that I am planning a distinction between these different categories is not very relevant. Moreover since I only use a subset of participants (5 out of a total of ~20), it is unlikely that given so little data we would be able to detect such fine distinction. Hence in this analysis we only care for:
1. Correctly answered trials
2. Go and NoGo trials
### Brief Task Description
In the task participants were given a set of cues which required either to give one of two responses (left and right) or withhold a response. The cues differed in that, if they were correctly answered they were followed by probablistic reward (80%) - wind cues, or the other type if incorrectly answered was followed by probablistic punishment (80%) - avoid cues.

Initially the goal of my analysis was to only look at correctly answered trials. However, upon inspecting the amount of correctly answered trials I became a bit warry about that goal. Since I only want to see the difference between Go and NoGo responses it might be feasible to just distinguish all trials answered as Go (no matter if correct or incorrect) from correctly answered NoGo trials. The rationale behind this reaoning is, that also incorrectly answered GoTrials reflect a decision that is made (albeit wrong) whereas an incorrectly answered NoGo trial likely reflects a lot of neural data where evidence for or against a decision was inconclusive. In order to see which approach may give me better results I will plot both types of data and perform the analysis on all Go responses. If I have enough time, I may perform the same analysis again just with the correctly answered Go responses.
