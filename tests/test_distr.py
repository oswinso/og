import numpy as np

from og.rl.distr import categorical_l2_project


def main():
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


if __name__ == "__main__":
    main()
