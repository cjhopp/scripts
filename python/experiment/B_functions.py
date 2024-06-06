import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

from scipy import linalg
from scipy.optimize import fsolve



def bigP(particle_velocity, Slope, Intercept, flyrho, flyer_velocity, sample_density, sample_us):
    """
    Explanation of what this thingy does

    :param particle_velocity:
    :param Slope:
    :param Intercept:
    :param flyrho:
    :param flyer_velocity:
    :param sample_density:
    :param sample_us:
    :return:
    """
    var = (Slope * particle_velocity**2 - (Intercept + 2 * Slope * flyer_velocity) *
           particle_velocity + Intercept * flyer_velocity + Slope * flyer_velocity**2)
    another_variable = sample_density * sample_us * particle_velocity
    return flyrho * var - another_variable


def MC_impmatch(steps, flyer_velocity, flyer_velocity_error,
                sample_us, sample_us_error, sample_density, sample_density_error):
    """
    Aluminum flyer parameters and Monte Carlo.

    :param steps:
    :param flyer_velocity:
    :param flyer_velocity_error:
    :param sample_us:
    :param sample_us_error:
    :param sample_density:
    :param sample_density_error:
    :return:
    """
    # C0, S
    Al_Hug_params = [1.188545520433363, 6.322038234532245]
    # covariance of Hugoniot parameters
    Al_Hug_cov = [[0.000419554462407, -0.004605429541366], [-0.004605429541366, 0.053580461575632]]

    Al_Hug_cov_lower = sp.linalg.cholesky(Al_Hug_cov,lower=True)

    Al_rho = 2700 #kg/m^3, density of the Al flyer

    Al_rho_error = Al_rho*0.003 # uncertainty
    # Set up Monte Carlo for samples

    root = np.ones(steps)
    upmc = np.ones(steps)
    rhomc = np.ones(steps)
    Pmc = np.ones(steps)

    for step in range(steps):
        temp_matrix = np.random.randn(2)
        bmat = np.matmul(Al_Hug_cov_lower, temp_matrix)
        S1 = Al_Hug_params[0] + bmat[0]
        C0 = Al_Hug_params[1] + bmat[1]

        flyer_velocity_mc = flyer_velocity + flyer_velocity_error * np.random.randn()

        flyer_rho_mc = Al_rho + Al_rho_error * np.random.randn()

        sample_rho_mc = sample_density + sample_density_error * np.random.randn()

        sample_us_mc = sample_us + sample_us_error * np.random.randn()

        print('{}\n{}\n{}\n{}\n{}\n{}'.format(S1, C0, flyer_rho_mc, flyer_velocity_mc, sample_rho_mc, sample_us_mc))

        '''
        This root function is the problem I think...
        '''
        root[step] = fsolve(bigP, 5, args=(S1, C0, flyer_rho_mc, flyer_velocity_mc, sample_rho_mc, sample_us_mc))

        upmc[step] = root[step]
        rhomc[step] = (sample_us_mc / (sample_us_mc - upmc[step])) * sample_rho_mc
        Pmc[step] = sample_rho_mc * sample_us_mc * upmc[step]

    up = np.mean(upmc)
    uperr = np.std(upmc)
    P = np.mean(Pmc) * 1e-3
    Perr = np.std(Pmc) * 1e-3
    rho = np.mean(rhomc) * 1e-3
    rhoerr = np.std(rhomc) * 1e-3

    return up, uperr, P, Perr, rho, rhoerr





'''

Here I'm trying to use the function. I opted to only use on variable parameter for the test, but in the future only the steps will be constant - the rest will come from the table I read in.

'''
if __name__ in '__main__':
    Zdata = pd.read_csv('Downloads/Z_data.csv')
    results = []
    for _, sample in Zdata.iterrows():
        result = MC_impmatch(3, sample['Flyer velocity'], sample['Flyer err'], sample['Sample Us'],
                             sample['Sample Us err'],3212,2)
        results.append(result)
    results = np.array(results)
    Zdata['up'] = results[:, 0]
    Zdata['up err'] = results[:, 1]
    Zdata['P'] = results[:, 2]
    Zdata['P err'] = results[:, 3]
    Zdata['rho'] = results[:, 4]
    Zdata['rho err'] = results[:, 5]
    print(Zdata)
