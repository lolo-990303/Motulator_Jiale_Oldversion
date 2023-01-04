# pylint: disable=C0103
"""
This script configures sensorless vector control for a 2.2-kW permanent-magnet
synchronous motor drive.

"""
# %%
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from control.common import SpeedCtrl
from control.sm.vector import CurrentRef, CurrentCtrl2DOFPI  # , CurrentCtrl
from control.sm.vector import VectorCtrl, Datalogger
from control.sm.vector import SensorlessVectorCtrl, SensorlessObserver
from control.sm.torque import TorqueCharacteristics
from helpers import Sequence  # , Step
from config.mdl_pmsm_2kW import mdl


# %%
@dataclass
class BaseValues:
    """
    This data class contains the base values computed from the rated values.
    These are used for plotting the results.

    """
    # pylint: disable=too-many-instance-attributes
    w: float = 2*np.pi*75
    i: float = np.sqrt(2)*4.3
    u: float = np.sqrt(2/3)*370
    p: int = 3
    psi: float = u/w
    P: float = 1.5*u*i
    Z: float = u/i
    L: float = Z/w
    tau: float = p*P/w


# %% Define the controller parameters
@dataclass
class CtrlParameters:
    """
    This data class contains parameters for the control system of a 2.2-kW
    permanent-magnet synchronous motor.

    """
    # pylint: disable=too-many-instance-attributes
    sensorless: bool = False
    # Sampling period
    #T_s: float = 250e-6
    T_s: float = 250e-6
    # Bandwidths
    alpha_c: float = 2*np.pi*200
    alpha_fw: float = 2*np.pi*20
    alpha_s: float = 2*np.pi*4
    # Maximum values
    tau_M_max: float = 2*14.6
    i_s_max: float = 1.5*np.sqrt(2)*5
    # Voltage margin
    k_u: float = .95
    # Nominal values
    u_dc_nom: float = 540
    w_nom: float = 2*np.pi*75
    # Motor parameter estimates
    R_s: float = 3.6
    L_d: float = .036
    L_q: float = .051
    psi_f: float = .545
    p: int = 3
    J: float = .015
    # Sensorless observer
    w_o: float = 2*np.pi*40  # Used only in the sensorless mode


# %%
base = BaseValues()
pars = CtrlParameters()
tq = TorqueCharacteristics(pars)

# %% Generate LUTs
mtpa = tq.mtpa_locus(i_s_max=pars.i_s_max)
lims = tq.mtpv_and_current_limits(i_s_max=pars.i_s_max)
# MTPA locus
pars.i_sd_mtpa = mtpa.i_sd_vs_tau_M
# Merged MTPV and current limits
pars.tau_M_lim = lims.tau_M_vs_abs_psi_s
pars.i_sd_lim = lims.i_sd_vs_tau_M

# %% Choose controller
speed_ctrl = SpeedCtrl(pars)
current_ref = CurrentRef(pars)
current_ctrl = CurrentCtrl2DOFPI(pars)
# current_ctrl = CurrentCtrl(pars)
datalog = Datalogger()

if pars.sensorless:
    observer = SensorlessObserver(pars)
    ctrl = SensorlessVectorCtrl(pars, speed_ctrl, current_ref, current_ctrl,
                                observer, datalog)
else:
    ctrl = VectorCtrl(pars, speed_ctrl, current_ref, current_ctrl, datalog)

print(ctrl)

# %% Profiles
# Speed reference
times = np.array([0, .5, 1, 1.5, 2, 2.5,  3, 3.5, 4])
#values = np.array([0,  0, 1,   1, 0,  -1, -1,   0, 0])*base.w
values = np.array([0,  0, 1,   1, 1,  1, 1,   1, 1])*base.w
mdl.speed_ref = Sequence(times, values)
# mdl.speed_ref = Step(.2, 2*base.w)
# External load torque
times = np.array([0, .5, .5, 3.5, 3.5, 4])
#values = np.array([0, 0, 1, 1, 0, 0])*14
values = np.array([0, 0, 1, 1, 1, 1])*14
mdl.mech.tau_L_ext = Sequence(times, values)
# Stop time of the simulation
mdl.t_stop = mdl.speed_ref.times[-1]
# mdl.t_stop = 2

# %% Print the profiles
print('\nProfiles')
print('--------')
print('Speed reference:')
with np.printoptions(precision=1, suppress=True):
    print('    {}'.format(mdl.speed_ref))
print('External load torque:')
with np.printoptions(precision=1, suppress=True):
    print('    {}'.format(mdl.mech.tau_L_ext))

# %% Plot torque characteristics
if __name__ == '__main__':
    # Plot loci and characteristics
    tq.plot_current_loci(pars.i_s_max, base)
    # tq.plot_flux_loci(pars.i_s_max, base)
    tq.plot_torque_flux(pars.i_s_max, base)
    tq.plot_torque_current(pars.i_s_max, base)
    # tq.plot_angle_torque(base.psi, base)
