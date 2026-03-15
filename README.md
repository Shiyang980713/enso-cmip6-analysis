# ENSO CMIP6 Diversity Analysis

Evaluates and compares CMIP6 model performance in simulating ENSO diversity (EP vs CP ENSO).

## Scientific Pipeline
1. Load HadISST (OBS) + CMIP6 model SST data (1980-2014)
2. Compute SST anomalies (remove monthly climatology)
3. Compute area-weighted Nino1+2 and Nino4 indices
4. EP ENSO: regress out Nino4 → EOF1 on residual SST anomaly
5. CP ENSO: regress out Nino1+2 → EOF1 on residual SST anomaly
6. Pattern correlation: model EOF vs HadISST EOF (area-weighted)
7. Produce: EOF maps, scatter plot (EP vs CP corr), bar charts (sorted)

## Models (29 CMIP6)
ACCESS-CM2, ACCESS-ESM1-5, BCC-ESM1, BCC-CSM2-MR, CAMS-CSM1-0, CAS-ESM2-0, CESM2, CIESM, CMCC-CM2-HR4, CMCC-CM2-SR5, CMCC-ESM2, CanESM5, FGOALS-f3-L, FGOALS-g3, FIO-ESM-2-0, GFDL-CM4, GFDL-ESM4, GISS-E2-2-G, GISS-E2-2-H, INM-CM5-0, IPSL-CM6A-LR-INCA, KACE-1-0-G, KIOST-ESM, MCM-UA-1-0, MIROC6, MPI-ESM-1-2-HAM, MPI-ESM1-2-LR, NESM3, NorCPM1

## Reference
HadISST reanalysis dataset
