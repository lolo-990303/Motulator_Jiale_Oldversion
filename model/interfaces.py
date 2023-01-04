# pylint: disable=C0103
"""
This module includes the solver functions as well as the models for PWM
carrier comparison and computational delay.

"""
import numpy as np
from scipy.integrate import solve_ivp
from helpers import abc2complex


# %%
def solve(mdl, d_abc, u_ref, u_dc, t_span, max_step=np.inf):
    """
    Solve the continuous-time model over t_span.

    Parameters
    ----------
    mdl : object
        Model to be simulated.
    d_abc : array_like of floats, shape (3,)
        Duty ratio references in the interval [0, 1].
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    max_step : float, optional
        Max step size of the solver. The default is inf.

    """

    # Common code
    def run_solver(t_span):
        # Skip possible zero time spans
        if t_span[-1] > t_span[0]:
            # Get initial values
            x0 = mdl.get_initial_values()
            # Integrate
            sol = solve_ivp(mdl.f, t_span, x0, max_step=max_step)
            # Set the new initial values (last points of the solution)
            t0_new, x0_new = t_span[-1], sol.y[:, -1]
            mdl.set_initial_values(t0_new, x0_new)
            # Data logger
            if mdl.datalog:
                mdl.datalog.save(mdl, sol)
            # Measuring the delta_uc
            if mdl.svpwm_3lv.enabled:
                mdl.converter.meas_delta_uc(phase_neutral, mdl, sol)

    if mdl.svpwm_2lv.enabled:
        # Sampling period
        T_s = t_span[-1] - t_span[0]
        # Compute the normalized switching spans and the corresponding states
        tn_sw, q_sw= mdl.svpwm_2lv(u_ref, u_dc)
        # Convert the normalized switching spans to seconds
        t_sw = t_span[0] + T_s * tn_sw
        # Loop over the switching time spans
        for i, t_sw_span in enumerate(t_sw):
            # Update the switching state vector (constant over the time span)
            mdl.q = abc2complex(q_sw[i])
            # Run the solver
            run_solver(t_sw_span)
    elif mdl.svpwm_3lv.enabled:
        # Sampling period
        T_s = t_span[-1] - t_span[0]
        # Compute the normalized switching spans and the corresponding states
        tn_sw, q_sw= mdl.svpwm_3lv(u_ref, u_dc, mdl.converter.delta_uc)
        # Convert the normalized switching spans to seconds
        t_sw = t_span[0] + T_s * tn_sw
        # Loop over the switching time spans
        for i, t_sw_span in enumerate(t_sw):
            # Update the switching state vector (constant over the time span)
            # Find which phase is on the state "0"
            phase_neutral = np.argwhere(q_sw[i] == 0)
            mdl.q = abc2complex(q_sw[i])
            # Run the solver
            run_solver(t_sw_span)
        # Logging the delta_uc
        mdl.converter.delta_ucs.append(mdl.converter.delta_uc)
    elif not mdl.pwm.enabled:
        # Update the duty ratio space vector (constant over the time span)
        mdl.q = abc2complex(d_abc)
        # Run the solver
        run_solver(t_span)
    else:
        # Sampling period
        T_s = t_span[-1] - t_span[0]
        # Compute the normalized switching spans and the corresponding states
        tn_sw, q_sw = mdl.pwm(d_abc)
        # Convert the normalized switching spans to seconds
        t_sw = t_span[0] + T_s * tn_sw
        # Loop over the switching time spans
        for i, t_sw_span in enumerate(t_sw):
            # Update the switching state vector (constant over the time span)
            mdl.q = q_sw[i]
            # Run the solver
            run_solver(t_sw_span)


# %%
class PWM:
    """
    This class implements carrier comparison of three-phase PWM. The switching
    instants and the switching states are explicitly and exactly computed from
    the duty ratios. The switching instants can be used in the ODE solver.

    """

    # pylint: disable=R0903
    def __init__(self, enabled=True, N=2 ** 12):
        """
        Parameters
        ----------
        N : int, optional
            Amount of PWM quantization levels. The default is 2**12.
        enabled : Boolean, optional
            PMW enabled. The default is True.

        """
        self.N = N
        self.falling_edge = True
        self.enabled = enabled
        if not enabled:
            self.desc = 'PWM model:\n    disabled\n'
        else:
            self.desc = ('PWM model:\n'
                         '    {} quantization levels\n').format(self.N)

    def __call__(self, d_abc):
        """
        Compute the normalized switching instants and the switching states.

        Parameters
        ----------
        d_abc : array_like of floats, shape (3,)
            Duty ratios in the range [0, 1].

        Returns
        -------
        tn_sw : ndarray, shape (4,2)
            Normalized switching instants,
            tn_sw = [0, t1, t2, t3, 1].
        q : complex ndarray, shape (4,)
            Switching state space vectors corresponding to the switching
            instants. For example, the switching state q[1] is applied
            at the interval tn_sw[1].

        Notes
        -----
        Switching instants t_sw split the sampling period T_s into
        four spans. No switching (e.g. da = 0 or da = 1) or simultaneous
        switching instants (e.g da == db) lead to zero spans, i.e.,
        t_sw[i] == t_sw[i].

        """
        # Quantize the duty ratios to N levels
        d_abc = np.round(self.N * np.asarray(d_abc)) / self.N
        # Initialize the normalized switching instant array
        tn_sw = np.zeros((4, 2))
        tn_sw[3, 1] = 1
        # Could be understood as a carrier comparison
        if self.falling_edge:
            # Normalized switching instants (zero crossing instants)
            tn_sw[1:4, 0] = np.sort(d_abc)
            tn_sw[0:3, 1] = tn_sw[1:4, 0]
            # Compute the switching state array
            q_abc = (tn_sw[:, 0] < d_abc[:, np.newaxis]).astype(int)
        else:
            # Rising edge
            tn_sw[1:4, 0] = np.sort(1 - d_abc)
            tn_sw[0:3, 1] = tn_sw[1:4, 0]
            q_abc = (tn_sw[:, 0] >= 1 - d_abc[:, np.newaxis]).astype(int)
        # Change the carrier direction for the next call
        self.falling_edge = not self.falling_edge
        # Switching state space vector
        q = abc2complex(q_abc)
        return tn_sw, q

    def __str__(self):
        return self.desc


# %%
class SVPWM_2LV:
    '''

    This module is to provide Space Vector PWM modulation method
    for two-level inverter.

    The method is based on a demo model "Lookup Table-Based PMSM" from Plexim.
    Url: https://www.plexim.com/sites/default/files/demo_models_categorized/plecs/look_up_table_based_pmsm.pdf

    The overmodulation technique used here is based on:
    Guohui Yin, Jianwu Luo, Jie Wang, Hongtao Wang. “Grapic over-modulation
    technique for space-vector PWM”. In: Small and Special Electrical Machines (2014).
    URL: https://kns.cnki.net/kcms/detail/detail.aspx?FileName=WTDJ201402018&DbName=CJFQ2014
    '''

    def __init__(self, enabled=True):

        '''

        All switching states in its sector are counterclockwise
        For each action, only one phase leg changes

        '''
        self.enabled = enabled
        self.alpha_gs = np.linspace(0, np.pi / 6, 1000, endpoint=True)
        self.M1 = np.sqrt((6 / np.pi) * (np.tan(np.pi / 6 - self.alpha_gs) + self.alpha_gs / ((np.cos(np.pi / 6 - self.alpha_gs)) ** 2)))
        self.M2 = np.sqrt((6 / np.pi) * (np.tan(np.pi / 6 - self.alpha_gs) + 4 * self.alpha_gs / 3))
        self.M1 = np.flipud(self.M1)
        self.sw_states = [
            [[0, 0, 0],
             [1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]],

            [[1, 1, 1],
             [1, 1, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 1, 1],
             [1, 1, 1]],

            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [1, 0, 1],
             [1, 1, 1]],

            [[1, 1, 1],
             [1, 0, 1],
             [1, 0, 0],
             [0, 0, 0]]]

    def __call__(self, u_ref, u_dc):
        """
        Parameters
        ----------
        u_dc : float
        Full dc link voltage

        u_ref : complex
        Reference voltage vector

        """
        u_alpha = u_ref.real
        u_beta = u_ref.imag
        u_ref_mag = np.sqrt(u_alpha ** 2 + u_beta ** 2)
        u_ref_ang = (np.angle(u_ref) + 2 * np.pi) % (2 * np.pi)  # Calculating the angels of U_ref (0-2pi)


        # Sector Selection
        sector = int(u_ref_ang / (np.pi / 3)) % 6  # 0-5

        # Sector Angle: Alpha
        alpha = u_ref_ang - sector * np.pi / 3

        # Effective time on two vector (Counterclockwise  a-b)
        # Overmodulation
        M = u_ref_mag / (u_dc / np.sqrt(3))
        if M > 1:
            if M > 1.05 :
                alpha_g = np.interp(M, self.M2, self.alpha_gs)
                if alpha <= alpha_g and alpha >= 0:
                    T_a = 1
                    T_b = 0
                    T_0 = 0
                elif alpha >= np.pi / 3 - alpha_g and alpha <= np.pi/3:
                    T_a = 0
                    T_b = 1
                    T_0 = 0
                else:
                    beta = np.pi/6*(alpha-alpha_g)/(np.pi/6-alpha_g)
                    T_a = np.sin(np.pi / 3 - beta) / np.sin(np.pi / 3 + beta)
                    T_b = np.sin(beta) / np.sin(np.pi / 3 + beta)
                    T_0 = 0
            else:
                alpha_g = np.interp(M, self.M1, self.alpha_gs)
                alpha_g = (1 - alpha_g / (np.pi / 6)) * (np.pi / 6)
                if alpha >= alpha_g and alpha <= np.pi / 3 - alpha_g:
                    T_a = np.sin(np.pi / 3 - alpha) / np.sin(np.pi / 3 + alpha)
                    T_b = np.sin(alpha) / np.sin(np.pi / 3 + alpha)
                    T_0 = 1 - T_a - T_b
                else:
                    u_ref_mag = 2/3*u_dc*np.sin(np.pi/3)/np.sin(2*np.pi/3-alpha_g)
                    T_a = np.sqrt(3) * np.sin(np.pi / 3 - alpha) * u_ref_mag / u_dc
                    T_b = np.sqrt(3) * np.sin(alpha) * u_ref_mag / u_dc
                    T_0 = 0
        else:
            T_a = np.sqrt(3) * np.sin(np.pi / 3 - alpha) * u_ref_mag / u_dc
            T_b = np.sqrt(3) * np.sin(alpha) * u_ref_mag / u_dc
            T_0 = 1 - T_a - T_b

        '''
        Calculating the switching sequence
        000-100-110-111-110-100-000
        000-010-110-111-110-010-000
        000-010-011-111-011-010-000
        000-001-011-111-011-001-000
        000-001-101-111-101-001-000
        000-100-101-111-101-100-000

        '''
        sw_sequ = np.zeros(7)
        sw_time = np.zeros(7)
        if (sector % 2 == 0):  # Sector 0 2 4
            sw_sequ[0] = 0
            sw_time[0] = 0 + 0.25 * T_0
            sw_sequ[1] = 1
            sw_time[1] = 0 + 0.25 * T_0 + 0.5 * T_a
            sw_sequ[2] = 2
            sw_time[2] = 0 + 0.25 * T_0 + 0.5 * T_a + 0.5 * T_b
            sw_sequ[3] = 3
            sw_time[3] = 0 + 1 - (0.25 * T_0 + 0.5 * T_a + 0.5 * T_b)
            sw_sequ[4] = 2
            sw_time[4] = 0 + 1 - (0.25 * T_0 + 0.5 * T_a)
            sw_sequ[5] = 1
            sw_time[5] = 0 + 1 - 0.25 * T_0
            sw_sequ[6] = 0
            sw_time[6] = 0 + 1
        else:  # Sector 1 3 5
            sw_sequ[0] = 3
            sw_time[0] = 0 + 0.25 * T_0
            sw_sequ[1] = 2
            sw_time[1] = 0 + 0.25 * T_0 + 0.5 * T_b
            sw_sequ[2] = 1
            sw_time[2] = 0 + 0.25 * T_0 + 0.5 * T_b + 0.5 * T_a
            sw_sequ[3] = 0
            sw_time[3] = 0 + 1 - (0.25 * T_0 + 0.5 * T_b + 0.5 * T_a)
            sw_sequ[4] = 1
            sw_time[4] = 0 + 1 - (0.25 * T_0 + 0.5 * T_b)
            sw_sequ[5] = 2
            sw_time[5] = 0 + 1 - 0.25 * T_0
            sw_sequ[6] = 3
            sw_time[6] = 0 + 1

        # Calculating switching states q
        qs = np.zeros((7,3))
        tn_sw = np.zeros((7, 2))
        tn_sw[1:7, 0] = sw_time[0:6]
        tn_sw[:, 1] = sw_time[:]
        for i in range(7):
            j=int(sw_sequ[i])
            qs[i] = self.sw_states[sector][j]
        return tn_sw, qs

class SVPWM_3LV:
    '''

    This module is to provide Space Vector PWM modulation method
    for three-level inverter.

    This method is based on the application report from Texas Instruments
    "Center-Aligned SVPWM Realization for 3- Phase 3- Level Inverter":
    https://www.ti.com/lit/pdf/sprabs6

    The overmodulation technique used here is based on:
    Bolognani, S. and Zigliotto, M. “Novel digital continuous control of SVM
    inverters in the overmodulation range”.
    DOI: 10.1109/28.568019.

    '''

    def __init__(self, enabled=True):
        """
        Parameters

        ----------
        enabled : Boolean, optional
            SVPMW enabled. The default is True.

        sw_states: ndarray, shape (6,4,3)
            Switching states of 6 sectors in 2-level SVPWM.
            The arrangement is counterclockwise and only one phase leg
            changes in each action, which minimizes the loss of inverters.

        """

        self.enabled = enabled
        self.sw_states = np.array([
            [[0, 0, 0],
             [1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]],

            [[1, 1, 1],
             [1, 1, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 1, 1],
             [1, 1, 1]],

            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [1, 0, 1],
             [1, 1, 1]],

            [[1, 1, 1],
             [1, 0, 1],
             [1, 0, 0],
             [0, 0, 0]]])

    def __call__(self, u_ref, u_dc, delta_uc):
        """
        Parameters
        ----------
        u_dc : float
        Full dc link voltage

        u_ref : complex
        Reference voltage vector

        """

        u_alpha = u_ref.real
        u_beta = u_ref.imag

        # Calculating the angels of u_ref (0-2pi)
        u_ref_ang = (np.arctan2(u_beta,u_alpha) + 2 * np.pi) % (2 * np.pi)

        # Determine the main sector and apply mapped reference voltage
        sub_hex_deter = int(u_ref_ang // (np.pi / 6))

        if sub_hex_deter == 0 or sub_hex_deter == 11:
            sub_hex = 1
            u_alpha_map = u_alpha - u_dc / 3
            u_beta_map = u_beta
        elif sub_hex_deter == 1 or sub_hex_deter == 2:
            sub_hex = 2
            u_alpha_map = u_alpha - u_dc / 6
            u_beta_map = u_beta - u_dc / 6 * np.sqrt(3)
        elif sub_hex_deter == 3 or sub_hex_deter == 4:
            sub_hex = 3
            u_alpha_map = u_alpha + u_dc / 6
            u_beta_map = u_beta - u_dc / 6 * np.sqrt(3)
        elif sub_hex_deter == 5 or sub_hex_deter == 6:
            sub_hex = 4
            u_alpha_map = u_alpha + u_dc / 3
            u_beta_map = u_beta
        elif sub_hex_deter == 7 or sub_hex_deter == 8:
            sub_hex = 5
            u_alpha_map = u_alpha + u_dc / 6
            u_beta_map = u_beta + u_dc / 6 * np.sqrt(3)
        elif sub_hex_deter == 9 or sub_hex_deter == 10:
            sub_hex = 6
            u_alpha_map = u_alpha - u_dc / 6
            u_beta_map = u_beta + u_dc / 6 * np.sqrt(3)
        else:
            sub_hex = 1
            u_alpha_map = 0
            u_beta_map = 0

        # Calculating the mapped voltage's magnitude and angle (0-2pi)
        u_map_mag = np.sqrt(u_alpha_map ** 2 + u_beta_map ** 2)
        u_map_ang = (np.arctan2(u_beta_map,u_alpha_map) + 2 * np.pi) % (2 * np.pi)

        # Sector Selection in two-level
        sector = int(u_map_ang / (np.pi / 3) % 6)  # 0-5
        # Sector Angle: theta
        theta = u_map_ang - sector * np.pi / 3

        # Overmodulation
        M = u_map_mag / (u_dc / (2 * np.sqrt(3)))
        if M > 1:
            if M > 2 / np.sqrt(3):
                if theta <= np.pi / 6:
                    T_a = 1
                    T_b = 0
                    T_0 = 0
                else:
                    T_a = 0
                    T_b = 1
                    T_0 = 0
            else:
                alpha_g = np.pi / 6 - np.mod(np.arccos(u_dc / (u_map_mag * 2 * np.sqrt(3))), 2 * np.pi)
                if 0 <= theta and theta <= alpha_g:
                    theta = theta
                elif alpha_g <= theta and theta <= np.pi / 6:
                    theta = alpha_g
                elif np.pi / 6 <= theta and theta <= np.pi / 3 - alpha_g:
                    theta = np.pi / 3 - alpha_g
                elif np.pi / 3 - alpha_g <= theta and theta <= np.pi / 3:
                    theta = theta
                T_a = np.sqrt(3) * u_map_mag / u_dc * 2 * (np.sqrt(3.0) / 2 * np.cos(theta) - 1.0 / 2.0 * np.sin(theta))
                T_b = np.sqrt(3) * u_map_mag / u_dc * 2 * np.sin(theta)
                T_0 = 1 - T_a - T_b
        else:
            T_a = np.sqrt(3) * u_map_mag / u_dc * 2 * (np.sqrt(3.0) / 2 * np.cos(theta) - 1.0 / 2.0 * np.sin(theta))
            T_b = np.sqrt(3) * u_map_mag / u_dc * 2 * np.sin(theta)
            T_0 = 1 - T_a - T_b

        sw_sequ = np.zeros(7)
        sw_time = np.zeros(7)

        # Neutral voltage point unbalanced problem
        # k=0 means close this function
        k=15
        if sector % 2 == 0:  # Sector 0 2 4
            sw_sequ[0] = 0
            sw_time[0] = 0 + 0.25 * T_0*(1-k*delta_uc/u_dc)
            sw_sequ[1] = 1
            sw_time[1] = 0 + 0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_a
            sw_sequ[2] = 2
            sw_time[2] = 0 + 0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_a + 0.5 * T_b
            sw_sequ[3] = 3
            sw_time[3] = 0 + 1 - (0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_a + 0.5 * T_b)
            sw_sequ[4] = 2
            sw_time[4] = 0 + 1 - (0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_a)
            sw_sequ[5] = 1
            sw_time[5] = 0 + 1 - 0.25 * T_0*(1-k*delta_uc/u_dc)
            sw_sequ[6] = 0
            sw_time[6] = 0 + 1
        else:  # Sector 1 3 5
            sw_sequ[0] = 3
            sw_time[0] = 0 + 0.25 * T_0*(1-k*delta_uc/u_dc)
            sw_sequ[1] = 2
            sw_time[1] = 0 + 0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_b
            sw_sequ[2] = 1
            sw_time[2] = 0 + 0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_b + 0.5 * T_a
            sw_sequ[3] = 0
            sw_time[3] = 0 + 1 - (0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_b + 0.5 * T_a)
            sw_sequ[4] = 1
            sw_time[4] = 0 + 1 - (0.25 * T_0*(1-k*delta_uc/u_dc) + 0.5 * T_b)
            sw_sequ[5] = 2
            sw_time[5] = 0 + 1 - 0.25 * T_0*(1-k*delta_uc/u_dc)
            sw_sequ[6] = 3
            sw_time[6] = 0 + 1

        # Replacing states and calculating switching states q
        if sub_hex == 1:
            a_state = 'PO'
            b_state = 'ON'
            c_state = 'ON'
        elif sub_hex == 2:
            a_state = 'PO'
            b_state = 'PO'
            c_state = 'ON'
        elif sub_hex == 3:
            a_state = 'ON'
            b_state = 'PO'
            c_state = 'ON'
        elif sub_hex == 4:
            a_state = 'ON'
            b_state = 'PO'
            c_state = 'PO'
        elif sub_hex == 5:
            a_state = 'ON'
            b_state = 'ON'
            c_state = 'PO'
        elif sub_hex == 6:
            a_state = 'PO'
            b_state = 'ON'
            c_state = 'PO'

        tn_sw = np.zeros((7, 2))
        tn_sw[1:7, 0] = sw_time[0:6]
        tn_sw[:, 1] = sw_time[:]
        qs = np.empty((7,3))
        q_abc = np.empty(3, dtype=int)
        for i in range(7):
            j = int(sw_sequ[i])
            q_abc[0] = self.sw_states[sector][j][0] if a_state == 'PO' else self.sw_states[sector][j][0] - 1
            q_abc[1] = self.sw_states[sector][j][1] if b_state == 'PO' else self.sw_states[sector][j][1] - 1
            q_abc[2] = self.sw_states[sector][j][2] if c_state == 'PO' else self.sw_states[sector][j][2] - 1
            qs[i] = q_abc/2
        return tn_sw, qs

# %%
class Delay:
    """
    This class implements a delay as a ring buffer.

    """

    # pylint: disable=R0903
    def __init__(self, length=1, elem=3):
        """
        Parameters
        ----------
        length : int, optional
            Length of the buffer in samples. The default is 1.

        """
        self.data = length * [elem * [0]]  # Creates a zero list
        self.desc = (('Computational delay:\n    {} sampling periods\n')
                     .format(length))

    def __call__(self, u):
        """
        Parameters
        ----------
        u : array_like, shape (elem,)
            Input array.

        Returns
        -------
        array_like, shape (elem,)
            Output array.

        """
        # Add the latest value to the end of the list
        self.data.append(u)
        # Pop the first element and return it
        return self.data.pop(0)

    def __str__(self):
        return self.desc
