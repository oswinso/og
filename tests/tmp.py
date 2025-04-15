from typing import Tuple

import jax
import jax.numpy as jnp


def calculate_gae_reach3(
    gamma: float,
    gae_lambda: float,
    T_hs: jnp.ndarray,
    T_Vhs: jnp.ndarray,
    done: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, hs, Vhs, done_row = inp
        next_Vhs_row, gae_coeffs = carry

        #                            *  *        *   *             *     *
        # Update GAE coeffs. [1] -> [λ 1-λ] -> [λ² λ(1-λ) 1-λ] -> [λ³ λ²(1-λ) λ(1-λ) 1-λ]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_lambda ** (ii + 1))
        gae_coeffs = gae_coeffs.at[1].set((gae_lambda ** ii) * (1 - gae_lambda))

        mask = jnp.arange(T + 1) < ii + 1
        mask_h = mask[:, None]

        # DP for Vh.
        done_row_processed = jnp.where(jnp.isnan(done_row * jnp.inf), 0, done_row * jnp.inf)
        disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row + done_row_processed)
        # disc_to_h = (1 - gamma) * hs + gamma * (next_Vhs_row)
        Vhs_row = jnp.minimum(hs, disc_to_h)
        Vhs_row = mask_h * Vhs_row

        Qhs_GAE = jnp.sum(Vhs_row * gae_coeffs, axis=0)

        # Setup Vs_row for next timestep.
        Vhs_row = jnp.roll(Vhs_row, 1, axis=0)
        Vhs_row = Vhs_row.at[0, :].set(Vhs)

        return (Vhs_row, gae_coeffs), Qhs_GAE

    done = jnp.array(done, dtype=int)
    init_gae_coeffs = jnp.zeros((T + 1, nh))

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(T_Vhs[T, :])
    init_carry = (init_Vhs, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_hs[:-1], T_Vhs[1:], done)

    _, Qhs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    return Qhs_GAEs - T_Vhs[:-1], Qhs_GAEs
