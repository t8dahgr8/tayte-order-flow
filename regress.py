"""OLS with naive and Newey-West standard errors side by side."""
import numpy as np
import statsmodels.api as sm


def fit_ols(df, ycol, xcol, maxlags):
    """Regress ycol on xcol. Returns both error estimates, same coefficient."""
    d = df[[ycol, xcol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 100:
        return None
    x = sm.add_constant(d[xcol].to_numpy())
    y = d[ycol].to_numpy()
    naive = sm.OLS(y, x).fit()
    #consecutive rows share almost their whole forward window, so residuals are
    #massively autocorrelated. HAC widens the errors to account for that.
    hac = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "n": len(d),
        "coef": float(naive.params[1]),
        "coef_bps": float(naive.params[1]) * 1e4,
        "se_naive": float(naive.bse[1]),
        "t_naive": float(naive.tvalues[1]),
        "se_hac": float(hac.bse[1]),
        "t_hac": float(hac.tvalues[1]),
        "r2": float(naive.rsquared),
    }


def print_table(rows, title):
    print(f"\n{title}")
    print(f"{'horizon':<10}{'n':>10}{'coef(bps)':>12}{'se_naive':>12}{'t_naive':>10}"
          f"{'se_HAC':>12}{'t_HAC':>10}{'R2':>10}")
    print("-" * 86)
    for label, r in rows:
        if r is None:
            print(f"{label:<10}{'too few rows':>10}")
            continue
        print(f"{label:<10}{r['n']:>10,}{r['coef_bps']:>12.3f}{r['se_naive'] * 1e4:>12.4f}"
              f"{r['t_naive']:>10.1f}{r['se_hac'] * 1e4:>12.4f}{r['t_hac']:>10.1f}"
              f"{r['r2']:>10.5f}")
