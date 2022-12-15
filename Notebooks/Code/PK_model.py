import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy import linalg
from collections.abc import Iterable
import pints


def central_dydt(current_state, params):
    per_conc = current_state["peripheral"]/params["Vol_perif"]
    dC_dt = (
        params["Q_dose"]*current_state["dose"]/params["Vol_dose"] +
        sum(params["Q_perif"]*per_conc) -
        sum(params["Q_perif"]*(current_state["central"]/params["Vol_C"])) -
        params["Cl"]*(current_state["central"]/params["Vol_C"])
        )
    return dC_dt


def perif_dydt(current_state, params):
    dP_dt = (
        params["Q_perif"]*(current_state["central"]/params["Vol_C"]) -
        params["Q_perif"]*(current_state["peripheral"]/params["Vol_perif"])
        )
    return dP_dt


def dose_dydt(current_state, params):
    dD_dt = -params["Q_dose"]*current_state["dose"]/params["Vol_dose"]
    return dD_dt


def dose_iv(t):
    return 0


def PK_dydt(t, y, params, comp=2, dose_comp=False):
    if dose_comp:
        current_state = {
            'central': y[0],
            "peripheral": np.asarray(y[1:-1]),
            "dose": y[-1]
        }
    else:
        current_state = {
            'central': y[0],
            "peripheral": np.asarray(y[1:]),
            "dose": dose_iv(t)
        }

    dC_dt = np.asarray([central_dydt(current_state, params)])
    if comp > 1:
        dP_dt = perif_dydt(current_state, params)
    else:
        dP_dt = np.asarray([])
    if dose_comp:
        dD_dt = np.asarray([dose_dydt(current_state, params)])
    else:
        dD_dt = np.asarray([])

    dy_dt = np.concatenate((dC_dt, dP_dt, dD_dt))

    return dy_dt


def PK_result(dose, num_comp, parameter, times, dose_comp=False, contin=False, tol=1e0):
    if len(parameter) != 2*(num_comp + int(dose_comp)):
        raise ValueError(
            'Expected ' + str(2*(num_comp + int(dose_comp))) +
            'parameters, recieved ' + str(len(parameter)) +
            ' parameters.'
        )
    params = {
        "Vol_C": parameter[0],
        "Cl": parameter[1],
        "Vol_perif": np.asarray(parameter[2:1+num_comp]),
        "Q_perif": np.asarray(parameter[1+num_comp:2*num_comp])
    }
    y_0 = [0]*(num_comp + int(dose_comp))
    if dose_comp:
        params["Vol_dose"] = parameter[-2]
        params["Q_dose"] = parameter[-1]
        y_0[-1] = dose
    else:
        params["Vol_dose"] = 1
        params["Q_dose"] = 1
        y_0[0] = dose

    if contin:
        t_span = (0, max(times))
        results_amt = integrate.solve_ivp(
            PK_dydt,
            t_span,
            y_0,
            args=(params, num_comp, dose_comp),
            dense_output=True,
            rtol = 1e-3*tol, 
            atol = 1e-6*tol 
        )
        return results_amt.sol
    else:
        results_amt = integrate.odeint(
            PK_dydt,
            y_0,
            times,
            args=(params, num_comp, dose_comp),
            tfirst=True
            )
        if dose_comp:
            results_conc = results_amt/np.concatenate((
                np.asarray([params["Vol_C"]]),
                params["Vol_perif"],
                np.asarray([params["Vol_dose"]])
            ))
        else:
            results_conc = results_amt/np.concatenate((
                np.asarray([params["Vol_C"]]),
                params["Vol_perif"]
            ))
        return results_conc


if __name__ == '__main__':
    # Params:
    k_dose = 1
    k_perif = 0.5
    Vol_perif = 1
    Cl = 0.5
    Vol_C = 2

    dose = 5
    time_span = 10
    times = np.linspace(0, time_span, 1000)
    plt.plot(
        times,
        PK_result(
            dose,
            3,
            [Vol_C, Cl, Vol_perif, Vol_perif*0.5, k_perif, k_perif+0.1],
            times
            )[:, :]
        )
    plt.show()


def solve_2_comp(dose, params, t):
    if isinstance(t, Iterable) is False and t < 0:
        return 0
    else:
        Vol_C = params[0]
        Cl = params[1]
        Vol_perif = params[2]
        Q = params[3]

        A = np.asarray([[-(Cl+Q)/Vol_C, Q/Vol_perif], [Q/Vol_C, -Q/Vol_perif]])
        x_0 = np.array([dose, 0])
        eigvalues, eigvectors = linalg.eig(A)

        if eigvalues[0] == eigvalues[1]:
            l = eigvalues[0]
            nu = eigvectors[0]
            rho = linalg.solve(A-l*np.identity(2), nu)
            c = linalg.solve([[nu[0], rho[0]], [nu[1], rho[1]]], x_0)
            C_t = (
                c[0]*np.exp(l*t)*nu[0] +
                c[1]*(t*np.exp(l*t)*nu[0] + np.exp(l*t)*rho[0])
                )

        elif eigvalues[0].imag == 0:
            l1, l2 = eigvalues
            nu1, nu2 = eigvectors
            c = linalg.solve([[nu1[0], nu2[0]], [nu1[1], nu2[1]]], x_0)
            C_t = c[0]*np.exp(l1*t)*nu1[0] + c[1]*np.exp(l2*t)*nu2[0]

        else:
            b = eigvalues[0].real
            a = eigvalues[0].imag
            nu = eigvectors[0]

            def u_v(time):
                ut_plus_ivt = (np.exp(b*time)*(
                    np.cos(a*time) + j*np.sin(b*time)
                )*nu)
                return (ut_plus_ivt.real, ut_plus_ivt.imag)

            u_0 = u_v(0)[0]
            v_0 = u_v(0)[1]
            c = linalg.solve([[u_0[0], v_0[0]], [u_0[1], v_0[1]]], x_0)
            C_t = c[0]*u_v(t)[0] + c[1]*u_v(t)[1]

        if isinstance(t, Iterable):
            times_before_dose = np.count_nonzero(t < 0)
            C_t[:times_before_dose] = 0
        return C_t.real/Vol_C


if __name__ == '__main__':

    Q_perif = 1
    Vol_perif = 1
    Cl = 1
    Vol_C = 2
    dose = 2

    times = np.linspace(-2, 48, 100)
    times_before_dose = np.count_nonzero(times < 0)
    int_times = np.concatenate((np.zeros(1), times[times_before_dose:]))

    plt.plot(
        times,
        solve_2_comp(dose, [Vol_C, Cl, Vol_perif, Q_perif], times),
        label='Analytical'
        )
    plt.plot(
        int_times,
        PK_result(
            dose, 2, [Vol_C, Cl, Vol_perif, Q_perif], int_times
            )[:, 0],
        label='ODEint'
        )
    plt.legend()
    plt.show()


class PintsPKLinIV(pints.ForwardModel):
    def __init__(
        self, data=None, patient_info=None, num_comp=2, dose_comp=False
    ):
        super(PintsPKLinIV, self).__init__()
        self.num_comp = num_comp
        if data is None:
            self.data = pd.DataFrame({'ID': [0]})
            self.single_patient = True
        else:
            self.data = data
            self.pseudotime = (
                data['ID'] *
                np.power(10, np.ceil(np.log10(data['TIME'].max())))
                + data['TIME']
            )
            self.single_patient = False

        if patient_info is None:
            dose_amt = self.data.drop_duplicates(['ID', 'DOSE'])
            dose_amt = dose_amt[['ID', 'DOSE']]
            self.dose = dict(dose_amt.values)
        elif isinstance(patient_info, pd.DataFrame):
            # dose_amt = patient_info[['ID', 'AMT']]
            dose_amt = patient_info.groupby(['DOSE'])['AMT'].mean()
            dose_amt = patient_info[['ID', 'DOSE']].join(
                dose_amt, on='DOSE', how='left'
            )
            self.dose = dict(dose_amt[['ID', 'AMT']].values)
        else:
            dose_amt = patient_info['AMT']
            ids = self.data.ID.unique()
            self.dose = dict(zip(ids, [dose_amt]*len(ids)))
        self.dose_comp = dose_comp
        # Things to help speed up the process
        if not self.single_patient:
            self._times = np.sort(self.data.TIME.unique())
            self._unique_doses = set(self.dose.values())
            self._results_masks = {}
            for patient_id in self.data.ID.unique():
                patient_times = self.data.loc[self.data['ID'] == patient_id]
                patient_times = patient_times[['TIME']]
                self._results_masks[patient_id] = np.isin(
                    self._times, patient_times
                )
        else:
            self._unique_doses = [dose_amt[0]]
            self._times = [0, 10]

    def n_parameters(self):
        return 2*(self.num_comp + int(self.dose_comp))

    def simulate(self, parameter, times):
        if self.single_patient:
            times = times
            return self.single_simulate(parameter, times, self.dose[0])
        else:
            results = {}
            for dose_amt in self._unique_doses:
                results[dose_amt] = np.asarray([
                    self._times,
                    self.single_simulate(parameter, self._times, dose_amt)
                ])
            all_result = np.asarray([0])
            for patient_id in self.data.ID.unique():
                indv_result = results[self.dose[patient_id]]
                indv_result = indv_result[1, :]
                indv_result = indv_result[self._results_masks[patient_id]]
                all_result = np.concatenate((all_result, indv_result[:]))
            return all_result[1:]

    def single_simulate(self, parameter, times, dose_amt):
        if times[0] == 0:
            result = PK_result(
                dose_amt,
                self.num_comp,
                parameter,
                times,
                dose_comp=self.dose_comp
            )[:, 0]
            return result
        elif times[0] > 0:
            # If the times do not start at 0 then odeint will begin simulation
            # at the first timepoint. However our model assumes dosing happens
            # at t=0 so we will need to add 0 to the times array and then
            # ignore the result at t=0.
            times = np.concatenate((np.array([0]), times))
            result = PK_result(
                dose_amt,
                self.num_comp,
                parameter,
                times,
                dose_comp=self.dose_comp
            )[1:, 0]
            return result
