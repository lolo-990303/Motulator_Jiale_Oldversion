# pylint: disable=C0103
"""
This script configures sensorless vector control for an induction motor.
The default values correspond to a 45-kW induction motor.

"""
# %%
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from control.common import SpeedCtrl
from control.im.vector import SensorlessObserver, CurrentModelEstimator
from control.im.vector import CurrentRef, CurrentCtrl2DOFPI  # , CurrentCtrl
from control.im.vector import SensorlessVectorCtrl, VectorCtrl, Datalogger
from helpers import Sequence  # , Step
from config.mdl_im_45kW import mdl


# %%
@dataclass
class BaseValues:
    """
    This data class contains the base values computed from the rated values.
    These are used for plotting the results.

    """
    # pylint: disable=too-many-instance-attributes
    w: float = 2*np.pi*50
    i: float = np.sqrt(2)*81
    u: float = np.sqrt(2/3)*400
    p: int = 2
    psi: float = u/w
    P: float = 1.5*u*i
    Z: float = u/i
    L: float = Z/w
    tau: float = p*P/w


# %% Define the controller parameters
@dataclass
class CtrlParameters:
    """
    This data class contains parameters for the control system.

    """
    # pylint: disable=too-many-instance-attributes
    sensorless: bool = True
    # Sampling period
    T_s: float = 2*250e-6
    # Bandwidths
    alpha_c: float = 2*np.pi*100
    alpha_o: float = 2*np.pi*40  # Used only in the sensorless mode
    alpha_s: float = 2*np.pi*1
    # Maximum values
    tau_M_max: float = 1.5*291
    i_s_max: float = 1.5*np.sqrt(2)*81
    # Nominal values
    psi_R_nom: float = .95
    u_dc_nom: float = 540
    # Motor parameter estimates
    R_s: float = .057
    R_R: float = .029
    L_sgm: float = 2.2e-3
    L_M: float = 24.5e-3
    p: int = 2
    J: float = 1.66*.49


# %%
base = BaseValues()
pars = CtrlParameters()

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
    observer = CurrentModelEstimator(pars)
    ctrl = VectorCtrl(pars, speed_ctrl, current_ref, current_ctrl,
                      observer, datalog)

print(ctrl)

# %% Profiles
# Speed reference
times = 5*np.array([0, .5, 1, 1.5, 2, 2.5,  3, 3.5, 4])
values = np.array([0,  0, 1,   1, 0,  -1, -1,   0, 0])*2*np.pi*50
mdl.speed_ref = Sequence(times, values)
# External load torque
times = np.array([0, .5, .5, 3.5, 3.5, 4])*5
values = np.array([0, 0, 1, 1, 0, 0])*291
mdl.mech.tau_L_ext = Sequence(times, values)  # tau_L_ext = Step(1, 14.6)
# Stop time of the simulation
mdl.t_stop = mdl.speed_ref.times[-1]

# %% Print the profiles
print('\nProfiles')
print('--------')
print('Speed reference:')
with np.printoptions(precision=1, suppress=True):
    print('    {}'.format(mdl.speed_ref))
print('External load torque:')
with np.printoptions(precision=1, suppress=True):
    print('    {}'.format(mdl.mech.tau_L_ext))
