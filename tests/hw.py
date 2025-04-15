import matplotlib.pyplot as plt
import numpy as np


def main():
    rng = np.random.default_rng(seed=12345)

    nx = 5
    s_n = [15, 50, 150]
    theta = np.ones(nx)

    n_iters = 10_000
    n_iters2 = 2_000

    for n in s_n:
        theta_hat_list = []
        for ii in range(n_iters):
            b_x = rng.uniform(-1, 1, size=(n, 5))
            b_eps = rng.standard_t(df=5, size=n)
            b_y = np.sum(b_x * theta, axis=-1) + b_eps

            # 1: Least squares estimate of theta from b_x and b_y.
            theta_hat = np.linalg.lstsq(b_x, b_y, rcond=None)[0]
            theta_hat_list.append(theta_hat)

        # Get the true variance of the first coordinate of theta_hat.
        theta_hat_list = np.stack(theta_hat_list, axis=0)
        true_variance = np.var(theta_hat_list[:, 0])
        print("[n={:4}] true variance: {}".format(n, true_variance))

        bootstrap_variances_list = []
        in_ci_list = []
        in_ci_list_pivotal = []
        for ii in range(n_iters2):
            b_x = rng.uniform(-1, 1, size=(n, 5))
            b_eps = rng.standard_t(df=5, size=n)
            b_y = np.sum(b_x * theta, axis=-1) + b_eps

            # 1: Least squares estimate of theta from b_x and b_y.
            theta_hat = np.linalg.lstsq(b_x, b_y, rcond=None)[0]

            # 2: Bootstrap estimate with B=200 samples, do a bootstrap estimate of the variance.
            B = 200
            theta_hat_b_list = []
            for b in range(B):
                idx = rng.choice(n, n, replace=True)
                b_x_b = b_x[idx]
                b_y_b = b_y[idx]
                theta_hat_b = np.linalg.lstsq(b_x_b, b_y_b, rcond=None)[0]
                theta_hat_b_list.append(theta_hat_b)

            theta_hat_b_list = np.stack(theta_hat_b_list, axis=0)

            # Variance of the first coordinate.
            bootstrap_variance = np.var(theta_hat_b_list[:, 0])
            bootstrap_variances_list.append(bootstrap_variance)

            # See what the coverage is of the bootstrap variance.
            bootstrap_std = np.sqrt(bootstrap_variance)
            ci_lo = theta_hat[0] - 1.64 * bootstrap_std
            ci_hi = theta_hat[0] + 1.64 * bootstrap_std
            in_ci = (theta[0] >= ci_lo) and (theta[0] <= ci_hi)
            in_ci_list.append(in_ci)

            # Do the same, but use the bootstrap pivotal confidence interval.
            ci_lo = 2 * theta_hat[0] - np.quantile(theta_hat_b_list[:, 0], 0.95)
            ci_hi = 2 * theta_hat[0] - np.quantile(theta_hat_b_list[:, 0], 0.05)
            in_ci = (theta[0] >= ci_lo) and (theta[0] <= ci_hi)
            in_ci_list_pivotal.append(in_ci)

        bootstrap_variances_list = np.array(bootstrap_variances_list)
        in_ci_list = np.array(in_ci_list)
        in_ci_list_pivotal = np.array(in_ci_list_pivotal)

        # Histogram plot of bootstrap estimate of variance, with true variance as a vertical line.
        fig, ax = plt.subplots()
        ax.hist(bootstrap_variances_list, bins=50, alpha=0.8, color="C1")
        ax.axvline(true_variance, color="C0")
        ax.set_title("n={}".format(n))
        fig.savefig("n={}.pdf".format(n), bbox_inches="tight")

        # Print the coverage.
        coverage = np.mean(in_ci_list)
        print("[n={:4}] coverage: {}".format(n, coverage))
        coverage = np.mean(in_ci_list_pivotal)
        print("[n={:4}] coverage: {} (pivotal)".format(n, coverage))


if __name__ == "__main__":
    main()
