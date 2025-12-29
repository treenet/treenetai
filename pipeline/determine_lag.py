
from lag_analysis_tool import estimate_lag, effective_sample_size, lag_significance
from lag_analysis_tool import plot_correlogram, plot_shift_overlay

local_rh = X[i, :, 1]     # channel 1
global_rh = X[i, :, 6]    # channel 6 (daily, broadcast)

lag, peak_corr, lags, corr = estimate_lag(local_rh, global_rh, max_lag=6*24)

# Effective sample size for significance test
Neff = effective_sample_size(local_rh, global_rh)
z, p = lag_significance(peak_corr, Neff)

print("Estimated lag:", lag)
print("Peak correlation:", peak_corr)
print("z-score:", z, "p-value:", p)

plot_correlogram(lags, corr, lag, title="Local RH vs Global RH")
plot_shift_overlay(local_rh, global_rh, lag)
