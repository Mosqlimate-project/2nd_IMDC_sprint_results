# 2nd Infodengue-Mosqlimate Dengue Challenge (IMDC): 2025 Sprint for dengue fever forecasts for Brazil

The Infodengue-Mosqlimate Dengue Challenge (IMDC) is an initiative led by the Mosqlimate and Infodengue in collaboration with the Harmonize and IDExtremes projects.

The objective of this 2025 sprint is **to promote training of predictive models and to develop high-quality ensemble forecast models for dengue in Brazil.**

The challenge involves three validation tests and one forecast target. The period of interest spans from the epidemiological week (EW) 41 of one year to EW 40 of the following year, aligning with the typical dengue season in Brazil.

**Validation test 1.** Predict the weekly number of dengue cases by state (UF) in the 2022-2023 season \[EW 41 2022- EW40 2023\], using data covering the period from EW 01 2010 to EW 25 2022;

**Validation test 2.** Predict the weekly number of dengue cases by state (UF) in the 2023-2024 season \[EW 41 2023- EW40 2024\], using data covering the period from EW 01 2010 to EW 25 2023;

**Validation test 3:** Predict the weekly number of dengue cases by state (UF) in the 2024-2025 season \[EW 41 2024- EW40 2025\], using data covering the period from EW 01 2010 to EW 25 2024;

**Forecast.** Predict the weekly number of dengue cases in Brazil, and by state (UF), in the 2025-2026 season \[EW 41 2025- EW40 2026\], using data covering the period from EW 01 2010 to EW 25 2025;

## Teams and models

In this 2nd edition, 15 teams contributed with 19 dengue forecast models for all Brazilian states for the years 2025 and 2026.

| Team/Model / Leader | Model ID | Approach | Spatial scale | Variables/datasets | Climate data |
|----------------------|----------|----------|---------------|--------------------|--------------|
| [Preditores_da_Picada](https://github.com/rick0110/Preditores_da_Picada) — Richard Elias Soares Viana (IMPA-Tech) | [108](https://api.mosqlimate.org/registry/model/108/) | SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) time series modeling | Municipality, State | Dengue cases, Temperature and humidity, Vector indices | Yes |
| [LaCiD/UFRN](https://github.com/lacidufrn/infodengue_sprint_2025) — Marcus Nunes ([LaCiD/UFRN](https://github.com/lacidufrn/infodengue_sprint_2025)) | [131*](https://api.mosqlimate.org/registry/model/131/) | ARIMAX (AutoRegressive Integrated Moving Average with eXogenous) | State | Temperature, Dengue cases | Yes |
| [JBD – Mosqlimate](https://github.com/davibarreira/jbd-mosqlimate-sprint) — Davi Sales Barreira (FGV/EMAp) | [133](https://api.mosqlimate.org/registry/model/133/) | Chronos (probabilistic time-series forecasting model from Amazon) | State | Dengue cases, Climate indices ENSO | Yes |
| [ISI Foundation](https://github.com/DavideNicola/ISI_Dengue_Model?tab=readme-ov-file) — Davide Nicola (ISI) | [134](https://api.mosqlimate.org/registry/model/134/) | A vector–host SEIR ODE system for humans and mosquitoes | State | Dengue cases, Weather data, Vector parameters | Yes |
| [The Global Health Resilience (GHR)](https://github.com/chlobular/ghr-imdc-2025) — Rachel Lowe (BSC) | [135](https://api.mosqlimate.org/registry/model/135/) | Bayesian hierarchical mixed-effects model | State, Health region | Dengue cases, Temperature, Precipitation, Surface temperature anomaly (ONI), Köppen, Biome | Yes |
| [Imperial College London](https://github.com/hadrianang/imperial-mosqlimate-sprint2025) — Hadrian Ang ([Imperial College London](https://github.com/hadrianang/imperial-mosqlimate-sprint2025)) | [136](https://api.mosqlimate.org/registry/model/136/) | Temporal Fusion Transformer (TFT), deep-learning + Random Forest for climate variables | State | Dengue cases, Temperature, Precipitation, Pressure, Relative humidity, Koppen climate classification, Brazilian biomes | Yes |
| [CERI Forecasting Club](https://github.com/graeme-dor/dengue-sprint-2025) — Graeme Dor (CERI Stellenbosch University) | [137](https://api.mosqlimate.org/registry/model/137/) | Ensemble: RF and LSTM per state, lowest RMSE chosen | State | Dengue cases, Temperature, Precipitation, Relative humidity | Yes |
| [TSMixer ZKI-PH4](https://github.com/DiogoParreira/ZKI-PH) — Diogo Parreira (Robert Koch Institute) | [138](https://api.mosqlimate.org/registry/model/138/) | Time Series Mixer (TSMixer) | Municipality, State | Dengue cases, Climate | Yes |
| [DengueSprint_Cornell-PEH](https://github.com/anabento/DengueSprint_Cornell-PEH) — Ana Bento (Cornell University) | [139](https://api.mosqlimate.org/registry/model/139/) | Negative Binomial Baseline Model | State | Dengue cases | No |
| [GeoHealth Dengue Forecasting Team](https://github.com/ChenXiang1998/2025-Infodengue-Sprint) — Paula Moraga (KAUST) | [141*](https://api.mosqlimate.org/registry/model/141/) | LSTM with climate covariates | State | Dengue cases, Temperature, Precipitation, Humidity, Pressure, Environmental data | Yes |
| [Strange Attractors Contributor](https://github.com/marciomacielbastos/MosqlimateSprint2025) — Marcio Maciel Bastos (FGV/EMAp) | [143](https://api.mosqlimate.org/registry/model/143/) | Bayesian state-level forecasting (Gravity Component + Bayesian Inference) | State | Dengue cases | No |
| [Beat it](https://github.com/lsbastos/sprint2025) — Leonardo Bastos (FIOCRUZ) | [144](https://api.mosqlimate.org/registry/model/144/) | Baseline Bayesian model — negative binomial with Gaussian random effects | State, Region | Dengue cases | No |
| [DS_OKSTATE](https://github.com/haridas-das/DS_OKSTATE_2025) — Lucas Storleman (Oklahoma State University) | [145](https://api.mosqlimate.org/registry/model/145/) | CNN–LSTM hybrid | Municipality, State | Dengue cases, Temperature, Precipitation, Humidity, Pressure, Environmental data | Yes |
| [D-FENSE/LNCC-AR_p-2025-1](https://github.com/americocunhajr/D-FENSE) — Americo Cunha Jr (LNCC / UERJ) | [150](https://api.mosqlimate.org/registry/model/150/) | AR(p) autoregressive process | State | Dengue cases, epiweek | No |
| [D-FENSE/UERJ-SARIMAX-2025-1](https://github.com/americocunhajr/D-FENSE) — Americo Cunha Jr (LNCC / UERJ) | [157](https://api.mosqlimate.org/registry/model/157/) | SARIMAX with exogenous inputs | State | Dengue cases, weekly temperature median, 52-week rolling mean of precipitation median  | Yes |
| [D-FENSE/LNCC-CliDENGO-2025-1](https://github.com/americocunhajr/D-FENSE) — Americo Cunha Jr (LNCC / UERJ) | [152](https://api.mosqlimate.org/registry/model/152/) | CLiDENGO (climate-modulated beta-logistic growth model) | State | Dengue cases, temperature (min/mean/max), precipitation (min/mean/max), and relative humidity (min/mean/max) | Yes |
| [D-FENSE/LNCC-SURGE-2025-1](https://github.com/americocunhajr/D-FENSE) — Americo Cunha Jr (LNCC / UERJ) | [154](https://api.mosqlimate.org/registry/model/154/) | SURGE (average surge model) | State | Dengue cases, epiweek | No |
| [Dengue oracle M1](https://github.com/eduardocorrearaujo/dengue-oracle) — Eduardo Araújo (FGV-EMAP) | [155](https://api.mosqlimate.org/registry/model/155/) | Baseline LSTM with cases, epiweek, population | Municipality, State, Health region | Dengue cases, epiweek, population | No |
| [Dengue oracle M2](https://github.com/eduardocorrearaujo/dengue-oracle) — Eduardo Araújo (FGV/EMAp) | [156](https://api.mosqlimate.org/registry/model/156/) | Baseline LSTM with covariates | Municipality, State, Health region | Dengue cases, epiweek, enso value, population, biome predominant | Yes |

\* Models 131 and 141 were not included in the validation results due to methodological or reproducibility issues

# Scoring and Ranking

The code used to generate the results below is available in the following notebooks:

* **`val_preds.ipynb`**: Validates whether the submitted predictions meet all the requirements.  
* **`load_the_predictions.ipynb`**: Loads the predictions submitted by all models and saves them into a CSV file.  
* **`log_normal_parameters.ipynb`**: Parametrizes each prediction for each date as a log-normal distribution.  
* **`compute_the_total_cases.ipynb`**: Computes the total predicted cases for each season.  
* **`compute_the_scores.ipynb`**: Calculates the WIS score for each model across all states and validation tests.  
* **`plot_scores.ipynb`**: Generates heatmap tables with WIS scores for each Brazilian region and highlights the best-performing models for both states and regions.  
* **`plot_series.ipynb`**: Plots the predicted time series for each model across states and validation tests.  


## Scores
Model performance was evaluated using the Weighted Interval Score (WIS), computed with the [mosqlient](https://github.com/Mosqlimate-project/mosqlimate-client/tree/main) Python package.
The WIS is a proper scoring rule for probabilistic forecasts
that balances sharpness and calibration. It summarizes forecast quality by doing a weighted average of the Interval Score (IS) and the absolute error of the median

The Interval Score for a central prediction interval with miscoverage rate $\alpha_k$ (i.e., coverage $1-\alpha_k$) is:

$$\textrm{IS}_{\alpha_k}(F, y) = (u_k - l_k) + \frac{2}{\alpha_k}(l_k - y) I\{y < l_k\} + \frac{2}{\alpha_k}(y - u_k) I\{y > u_k\},$$

where
- $F$ is the forecast cumulative distribution function (CDF)
- $y$ is the observed value
- $\alpha_k$ is the miscoverage rate of $F$ for the $k$-th interval
- $l_k$, $u_k$ are the $1 - \alpha_k/2$ and $1 + \alpha_k/2$ quantiles of $F$, respectively
- $I\{\cdot\}$ is the indicator function

The WIS is then computed as:


$$\textrm{WIS}_{\alpha_A}(F, y)=$$

$$\frac{1}{K + 1/2} \left( w_0 |y - m| + \sum_{k=1}^K w_k \textrm{IS}_{\alpha_k}(F, y) \right),$$

where
- $m$ is the predicted median
- A is the interval $[0:K]$
- $K$ is the number of prediction intervals
- $w_0$ is the weight for the median, which is set to 1/2 by default
- $w_k$ is the weight for the $k$-th interval, which is set to $\alpha_k/2$ by default


Each model in this challenge was evaluated by computing the average WIS for each state and validation test. We assessed model performance using the average WIS over the full period of each validation test (1, 2, and 3).


| Average Score S* | Validation test | Score (S) used | Evaluated range |
| -----------------| ---------------|-----------------| -----------------|
|𝑆<sub>1</sub> | 1 |WIS |EW41 2022 - EW40 2023  |
|𝑆<sub>2</sub> | 2 |WIS |EW41 2023 - EW40 2024  |
|𝑆<sub>3</sub> | 3 |WIS |EW41 2024 - EW25 2025  |
 
## Ranking

### Best-performing models per state

To rank the models within each state, we computed the average WIS across the entire evaluation period, which includes the three validation sets (EW41 2022 – EW25 2025). The model with the lowest average WIS was assigned rank 1, while the model with the highest average WIS was assigned rank 17, with the others ranked accordingly in between.

The bar plot below shows the number of states that each model achieved the best rank in.

![Best models by state](./figures/count_best_models_state.png)

The map below shows the Brazilian states colored according to the model that performed best in each. The state of Espírito Santo (ES) is left uncolored because it was excluded from the analysis.

![Map best models by state](./figures/map_best_model.png)

### Best-performing models per region

The ranking for each region was calculated using the following equation:

$$
R_{M} = \sum_{n=1}^{N} \frac{1}{R_{n}},
$$

where $n$ indexes the states within the region, $N$ is the total number of states in that region, and $M$ denotes the region as a whole. The state-level ranks $R_{n}$ were computed as described in the previous section. This formulation assigns greater weight to models that ranked among the top performers in individual states, ensuring that consistently high-performing models have a stronger influence on the regional ranking.

The table below presents the top five best-performing models in each region:

| Rank | North            | Northeast       | Midwest           | Southeast          | South              |
|------|------------------|-----------------|-------------------|--------------------|--------------------|
| 1    | LNN-AR_p-1       | GHR Model       | GHR Model         | Dengue oracle M2   | Dengue Oracle M1   |
| 2    | Beat it          | LNN-AR_p-1      | LSTM-RF model     | LNCC-SURGE-1       | LSTM-RF model      |
| 3    | Imperial-TFT Model | Cornell PEH   | Dengue oracle M2  | Dengue oracle M1   | LNN-AR_p-1         |
| 4    | Dengue Oracle M1 | Imperial-TFT Model | UERJ-SARIMAX-2 | GHR Model          | Beat it            |
| 5    | Cornell PEH      | CNNLSTM         | Chronos-Bolt      | UERJ-SARIMAX-2     | UERJ-SARIMAX-2     |


## WIS scores by region

The figures below present the WIS scores for each validation test and each state (x-axis) and model (y-axis). **The models are ordered from the best region model to the worst**. The green rectangle highlights the lowest WIS score in each column.

### North

![WIS north](./figures/all_models_WIS_north.png)

### Northeast

![WIS northeast](./figures/all_models_WIS_northeast.png)

### Midwest

![WIS midwest](./figures/all_models_WIS_midwest.png)


### Southeast

![WIS southeast](./figures/all_models_WIS_southeast.png)


### South

![WIS south](./figures/all_models_WIS_south.png)


## Predicted curves

The figures below show the curves of the submitted predictions by state. The model ID corresponding to each line is shown in the legend. The y-axis range is set to 1.5 times the maximum value of the observed data. Use the [Mosqlimate dashboard](https://api.mosqlimate.org/vis/dashboard/?dashboard=sprint
) to compare a subset of predictions and visualize their scores.

### North 

![preds AC](./figures/preds_AM.png)

![preds AP](./figures/preds_AP.png)

![preds AM](./figures/preds_AM.png)

![preds PA](./figures/preds_PA.png)

![preds RO](./figures/preds_RO.png)

![preds RR](./figures/preds_RR.png)

![preds TO](./figures/preds_TO.png)

### Northeast 

![preds AL](./figures/preds_AL.png)

![preds BA](./figures/preds_BA.png)

![preds CE](./figures/preds_CE.png)

![preds MA](./figures/preds_MA.png)

![preds PB](./figures/preds_PB.png)

![preds PE](./figures/preds_PE.png)

![preds PI](./figures/preds_PI.png)

![preds RN](./figures/preds_RN.png)

![preds SE](./figures/preds_SE.png)

### Midwest 

![preds DF](./figures/preds_DF.png)

![preds GO](./figures/preds_GO.png)

![preds MT](./figures/preds_MT.png)

![preds MS](./figures/preds_MS.png)

### Southeast 

![preds MG](./figures/preds_MG.png)

![preds RJ](./figures/preds_RJ.png)

![preds SP](./figures/preds_SP.png)

### South

![preds PR](./figures/preds_PR.png)

![preds RS](./figures/preds_RS.png)

![preds SC](./figures/preds_SC.png)



## Additional Analysis: Models' performance in predicting total cases 

To compute the total number of cases for each season based on the submitted predictions, we applied the following steps:  

1. **Weekly approximation**: For each predicted week in the season, we approximated the distribution as log-normal by fitting the submitted prediction intervals to the CDF of a log-normal distribution through an optimization procedure. To compute the parameters, we used the following procedure. Let $L$ denote the number of symmetric prediction intervals available for a given model, and assume that the median $m$ is always provided. In this case, one obtains $J = 2L + 1$ quantiles from the predictive distribution, resulting in a sequence of quantiles $q_j$ with associated probability levels $\gamma_j$.

To estimate the parameters $\theta$ of a log-normal distribution that best fit this sequence of quantiles, we solve the following optimization problem:

$$
\theta = \argmin_{\theta \in \boldsymbol{\Theta}} \sum_{j=1}^{J} \frac{|q_j - Q_\theta(\gamma_j)|}{|q_j|},
$$

where $Q_\theta(\gamma_j)$ denotes the $\gamma_j$-quantile of the log-normal distribution with parameters $\theta$.

Quantiles equal to zero are excluded from the optimization when they correspond to percentiles below the median (50th percentile). If the median itself is zero, we fix $\mu = 0.01$ and $\sigma = 0.5$ as as the parameters of the distribution.

2. **Sampling**: Using the $\mu$ and $\sigma$ parameters obtained for each week, we generated 1,000 samples from the log-normal distribution of that week. These weekly samples were then summed across all weeks, resulting in a final array of 1,000 samples representing the total cases for the season.

3. **Prediction intervals**: From the summed values, we calculated the 50%, 80%, 90%, and 95% prediction intervals, along with the median, thus obtaining a probabilistic estimate of the total cases for the season. In addition, we plotted the 95% prediction interval and the median against the observed values for each season and state. 

4. **Evaluation**: Based on these probabilistic predictions, we compared the estimated total cases with the observed totals and computed the WIS.  

The steps were implemented across different notebooks:  
- Step 1: `log_normal_parameters_CDF.ipynb`  
- Steps 2–3: `compute_the_total_cases.ipynb`  
- Step 4: (WIS plots): `plot_scores.ipynb`  


###  Best-performing models per region

The best-performing models were identified by ranking each model according to their WIS in the total cases prediction task, following the same procedure as described in the previous sections. The table below shows the top five models for each region:

|Rank | North | Northeast | Midwest | Southeast | South |
|-----| ------| ----------| --------| ----------| ------|
|1    | UERJ-SARIMAX-2   | Cornell PEH       | LSTM-RF model     | Dengue oracle M2       | UERJ-SARIMAX-2   |
|2    | LNN-AR_p-1   | CNNLSTM       | GHR Model     | IMPA-TECH       | LSTM-RF model   |
|3    | LNCC-SURGE-1   | LNCC-SURGE-1       | Cornell PEH     | Dengue Oracle M1       | ISI_Dengue_Model   |
|4    | Model fourier-gravidade   | LNCC-CLIDENGO-1       | Beat it     | Model fourier-gravidade       | Dengue oracle M2   |
|5    | Cornell PEH   | TSMixer ZKI-PH4       | UERJ-SARIMAX-2     | LSTM-RF model    | Imperial-TFT Model   |


## WIS scores by region

The figures below present the WIS scores for each validation test and each state (x-axis) and model (y-axis) concerning the total number of cases in the season. The models are ordered from the best regional model to the worst. The green rectangle highlights the lowest WIS score in each column.


### North

![WIS north](./figures/all_models_total_cases_WIS_north.png)

### Northeast

![WIS northeast](./figures/all_models_total_cases_WIS_northeast.png)

### Midwest

![WIS midwest](./figures/all_models_total_cases_WIS_midwest.png)


### Southeast

![WIS southeast](./figures/all_models_total_cases_WIS_southeast.png)


### South

![WIS south](./figures/all_models_total_cases_WIS_south.png)


### Plots

The figures below display the total number of cases predicted by each model using a parametric approximation to a log-normal distribution. The error bars represent the 95% prediction intervals. Red points and bars indicate models whose 95% prediction intervals did not capture the observed number of cases, while green points and bars indicate models whose intervals did include the observed values.

### North 

![preds AC](./figures/total_cases_AM.png)

![preds AP](./figures/total_cases_AP.png)

![preds AM](./figures/total_cases_AM.png)

![preds PA](./figures/total_cases_PA.png)

![preds RO](./figures/total_cases_RO.png)

![preds RR](./figures/total_cases_RR.png)

![preds TO](./figures/total_cases_TO.png)

### Northeast 

![preds AL](./figures/total_cases_AL.png)

![preds BA](./figures/total_cases_BA.png)

![preds CE](./figures/total_cases_CE.png)

![preds MA](./figures/total_cases_MA.png)

![preds PB](./figures/total_cases_PB.png)

![preds PE](./figures/total_cases_PE.png)

![preds PI](./figures/total_cases_PI.png)

![preds RN](./figures/total_cases_RN.png)

![preds SE](./figures/total_cases_SE.png)

### Midwest 

![preds DF](./figures/total_cases_DF.png)

![preds GO](./figures/total_cases_GO.png)

![preds MT](./figures/total_cases_MT.png)

![preds MS](./figures/total_cases_MS.png)

### Southeast 

![preds MG](./figures/total_cases_MG.png)

![preds RJ](./figures/total_cases_RJ.png)

![preds SP](./figures/total_cases_SP.png)

### South

![preds PR](./figures/total_cases_PR.png)

![preds RS](./figures/total_cases_RS.png)

![preds SC](./figures/total_cases_SC.png)


