# Advanced Data Analysis and Visualization in Logistics

Week 3 task: exploratory data analysis (EDA) and visualization of a simulated logistics dataset (2,000 shipments) using Python.

## What this project does
- Simulates shipment data (mode, region, carrier, distance, weight, transit time, delay, cost)
- Cleans data (median imputation, IQR outlier check)
- Computes descriptive statistics, correlations, a regression, a Welch t-test and an ANOVA
- Produces 8 visualizations (histogram+KDE, box plot, scatter, correlation heatmap, monthly trend, region x mode heatmap, carrier bars, violin plot)

## Key findings (simulated data)
- On-time delivery: 92.2% (Jan-Sep) vs 58.2% (Oct-Dec peak)
- Air costs about 17x more per kg than road
- Road to East region: only 56.8% on-time (bottleneck lane)
- Distance explains only about 13% of cost variance when all modes are pooled

## Project structure
```
src/analysis.py     # data simulation, EDA, figures
data/               # generated dataset (CSV)
figures/            # 8 PNG charts
outputs/            # descriptive stats and summary JSON
report/             # full Word report
requirements.txt
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/analysis.py
```
Run from the repository root. A fixed random seed (42) makes results reproducible.

## Note
The data is synthetic. The seasonal slowdown and East-road bottleneck were built into the simulation, so the analysis shows that they can be detected, not that they exist in a real network.
