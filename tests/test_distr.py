import ipdb
import jax
import numpy as np

from og.rl.distr import categorical_cvar, categorical_l2_project, midpoint_to_endpoints


def test_categorical_l2_project():
    rng = np.random.default_rng(seed=141293)

    for ii in range(16):
        n = rng.integers(7, 13)
        m = rng.integers(7, 13)

        n_probs = rng.uniform(size=n)
        n_zp = 0.9 * np.linspace(0.3, 0.8, n)

        m_zq = np.linspace(0.0, 1.0, m)

        m_probs = np.array(categorical_l2_project(n_zp, n_probs, m_zq))

        p_mean = np.dot(n_zp, n_probs)
        q_mean = np.dot(m_zq, m_probs)
        np.testing.assert_allclose(p_mean, q_mean)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.bar(n_zp, n_probs, width=n_zp[1] - n_zp[0], align="center", alpha=0.5, label="p")
    # ax.bar(m_zq, m_probs, width=m_zq[1] - m_zq[0], align="center", alpha=0.5, label="q")
    # ax.legend()
    # plt.show()


def test_categorical_cvar():
    rng = np.random.default_rng(seed=141293)

    # CVaR(0) should just be the mean.
    for ii in range(16):
        n = rng.integers(7, 13)
        n_zp = np.arange(n)
        n_probs = rng.uniform(size=n)
        n_probs /= n_probs.sum()

        mean = np.dot(n_zp, n_probs)
        mean_cvar = categorical_cvar(0.0, n_zp, n_probs)

        np.testing.assert_allclose(mean, mean_cvar)

    # import matplotlib.pyplot as plt
    #
    # n = 11
    # n_zp = np.arange(n)
    # n_probs = rng.uniform(size=n)
    # n_probs[:2] = 0.0
    # n_probs /= n_probs.sum()
    #
    # alpha = 0.005
    #
    # mean = np.dot(n_zp, n_probs)
    # print("mean: {}".format(mean))
    # mean_cvar = categorical_cvar(0.0, n_zp, n_probs)
    # np.testing.assert_allclose(mean, mean_cvar)
    #
    # n_alphas = 128
    # b_alphas = np.linspace(0.0, 1.0, n_alphas + 1)[:-1]
    # b_cvars = np.array(jax.vmap(lambda alpha: categorical_cvar(alpha, n_zp, n_probs))(b_alphas))
    #
    # np1_prob_cumsum = np.concatenate([np.array([0.0]), np.cumsum(n_probs)])
    # np1_endpoints = np.array(midpoint_to_endpoints(n_zp))
    #
    # print(n_zp)
    # print(np1_endpoints)
    #
    # fig, axes = plt.subplots(3)
    # axes[0].plot(np.cumsum(n_probs), n_zp, marker="o", lw=0.4, label="CDF")
    # axes[0].axvline(alpha)
    # axes[0].set_title("CDF midpoint")
    #
    # axes[1].plot(np1_prob_cumsum, np1_endpoints, marker="o", lw=0.4, label="CDF")
    # axes[1].axvline(alpha)
    # axes[1].set_title("CDF endpoint")
    #
    # axes[2].axhline(mean)
    # axes[2].plot(b_alphas, b_cvars, marker="o", lw=0.4, label="CVaR")
    # axes[2].axvline(alpha)
    # plt.show()


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    with ipdb.launch_ipdb_on_exception():
        test_categorical_l2_project()
        test_categorical_cvar()
