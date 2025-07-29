import pymc as pm
import numpy as np
import pandas as pd

def recover_parameters(df_r, random_seed=None):
    """Inference of parameter values for a simulated dataset.

    Args:
        df_r (pd.dataFrame): Pandas dataFrame as observations for the model.
        random_seed (int, optional): For reproducible results. Defaults to None.

    Returns:
        idata (az.): _description_
    """

    n_participants = len(df_r.participant.unique())
    n_trials = len(df_r.trial.unique())

    with pm.Model() as m_recover:
        rho = pm.HalfNormal("rho", sigma=10, shape=n_participants, dims=("participant"))
        beta = pm.HalfNormal(
            "beta", sigma=10, shape=n_participants, dims=("participant")
        )

        intended = pm.Normal.dist(
            mu=rho[df_r["participant"]], sigma=beta[df_r["participant"]], shape=n_trials
        )

        pump = pm.Censored(
            "pump",
            intended,
            lower=None,
            upper=df_r["burst"],
            observed=df_r["y"],
        )

        idata = pm.sample(target_accept=0.95, random_seed=random_seed)

    return idata


def simulate_data(
    mu_rho,
    sigma_rho,
    mu_beta,
    sigma_beta,
    n_participants,
    p_burst,
    n_trials,
    random_seed=None,
):
    """Simulates BART data for censored model.

    Args:
        mu_rho (float): Prior distribution parameter setting.
        sigma_rho (float): Prior distribution parameter setting.
        mu_beta (float): Prior distribution parameter setting.
        sigma_beta (float): Prior distribution parameter setting.
        n_participants (int): Number of participants.
        p_burst (float): Burst probability.
        n_trials (int): Number of trials.
        random_seed (int, optional): Random seed for simulations. Defaults to None.

    Returns:
        _type_: _description_
    """
    coords = {
        "participant": np.arange(n_participants),
        "trial": np.arange(n_trials),
    }

    with pm.Model(coords=coords) as m_parameter_recovery_simulate:
        rho = pm.TruncatedNormal(
            "rho",
            mu_rho,
            sigma_rho,
            lower=0,
            shape=n_participants,
            dims=("participant"),
        )
        beta = pm.TruncatedNormal(
            "beta",
            mu_beta,
            sigma_beta,
            lower=0,
            shape=n_participants,
            dims=("participant"),
        )

        intended = pm.TruncatedNormal.dist(
            mu=rho[:, None], sigma=beta[:, None], lower=0
        )
        pump_until_bursts = pm.Geometric.dist(p=p_burst)
        trial_bursts = pm.draw(pump_until_bursts, draws=n_trials)
        experiment = np.tile(trial_bursts, (n_participants, 1))

        pump = pm.Censored(
            "pump",
            intended,
            lower=None,
            upper=experiment,
            dims=("participant", "trial"),
        )

        idata = pm.sample_prior_predictive(samples=1, random_seed=random_seed)

    g = pm.model_to_graphviz(m_parameter_recovery_simulate)

    experiment = experiment[0, :]
    df_pr = pd.DataFrame(columns=["participant", "trial", "y", "burst"])
    df_pr["participant"] = np.repeat(
        idata.prior.participant.values, idata.prior.trial.size
    )
    df_pr["trial"] = np.tile(idata.prior.trial.values, idata.prior.participant.size)
    df_pr["burst"] = np.tile(experiment, idata.prior.participant.size).astype("float")
    df_pr["y"] = np.squeeze(idata.prior.pump.values).flatten().round()

    df_param_vals = pd.DataFrame(columns=["participant", "rho", "beta"])
    df_param_vals["participant"] = idata.prior.participant.values
    df_param_vals["rho"] = idata.prior.rho.values.flatten()
    df_param_vals["beta"] = idata.prior.beta.values.flatten()

    return df_pr, df_param_vals
