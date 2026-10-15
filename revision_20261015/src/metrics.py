import numpy as np

EPS = 1e-9

def score(y, pred, steps, test, observed):
    y, pred, steps = np.asarray(y, float), np.asarray(pred, float), np.asarray(steps, float)
    test, observed = np.asarray(test, int), np.asarray(observed, int)
    n = len(y)
    k = max(2, int(np.ceil(0.2 * n)))
    tail = np.unique(np.r_[np.arange(k), np.arange(n-k, n)])
    tail_test = np.intersect1d(tail, test)
    tail_unobserved = np.setdiff1d(tail, observed)
    peak = max(float(y.max()), EPS)
    error = pred[test] - y[test]
    sse = float(error @ error)
    sst = float(np.sum((y[test]-np.mean(y[test]))**2))
    rmse = lambda idx: float(100*np.sqrt(np.mean((pred[idx]-y[idx])**2))/peak) if len(idx) else np.nan
    def sym_kl(idx):
        if not len(idx):
            return np.nan
        p, q = np.maximum(y[idx], 0)+EPS, np.maximum(pred[idx], 0)+EPS
        p, q = p/p.sum(), q/q.sum()
        return float(0.5*np.sum(p*np.log(p/q)+q*np.log(q/p)))
    return dict(test_r2=1-sse/sst if sst > EPS else np.nan,
                test_nrmse_pct_peak=rmse(test),
                test_smape_pct=float(200*np.mean(np.abs(error)/np.maximum(np.abs(y[test])+np.abs(pred[test]), EPS))),
                tail_full_rmse_pct_peak=rmse(tail),
                tail_unobserved_rmse_pct_peak=rmse(tail_unobserved),
                tail_test_rmse_pct_peak=rmse(tail_test), tail_test_symmetric_kl=sym_kl(tail_test),
                peak_error_steps=float(abs(steps[np.argmax(y)]-steps[np.argmax(pred)])),
                tail_slope_rmse=float(np.sqrt(np.mean((np.gradient(pred, steps)[tail]-np.gradient(y, steps)[tail])**2))),
                test_n=len(test), tail_test_n=len(tail_test), observed_n=len(observed),
                sse=sse, sum_y=float(y[test].sum()), sum_y2=float(np.sum(y[test]**2)))
