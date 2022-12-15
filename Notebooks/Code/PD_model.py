import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import pints
from Code.PK_model import PK_result


class PD_model():
    def __init__(self):
        self.dose_amt = None
        self.PK_params = None
        self.PD_params = None
        self.model = None
        self.num_tr = 0
        self.num_comp = 0
        self.drug_amt = None
        self.y_0 = None
        self.tolerance = 1e-3

    def create_PK(self, dose_amt, PK_params, time_max, num_comp=2):
        # self.dose_amt = dose_amt
        # self.num_comp = num_comp
        self.PK_params = PK_params
        self.drug_amt = PK_result(
            dose_amt, num_comp, PK_params, [time_max], contin=True, tol=self.tolerance
        )

    def set_PD_model(self, model, PD_params, E_max=False, num_tr=3):

        if model == "Danyeka":
            # PD_params = ["R_0", "k_out", ...]
            if len(PD_params) != 3+int(E_max):
                raise ValueError(
                    'Expected ' + str(3+int(E_max)) +
                    ' parameters, but recieved ' +
                    str(len(PD_params)) + ' Parameters.'
                )
            self.num_tr = 0
            self.PD_params = {
                "k_in": PD_params[0]*PD_params[1], "k_out": PD_params[1]
            }
            self.y_0 = [PD_params[0]]

        elif model == "Zamboni":
            # PD_params = ["R_0", "MTT", "k_out", ...]
            if len(PD_params) != 4+int(E_max):
                raise ValueError(
                    'Expected ' + str(4+int(E_max)) +
                    ' parameters, but recieved ' +
                    str(len(PD_params)) + ' Parameters.'
                )
            self.num_tr = num_tr
            self.PD_params = {
                "k_in": PD_params[0]*PD_params[2],
                "k_out": PD_params[2],
                "k_tr": (num_tr+1)/PD_params[1]
            }
            self.y_0 = [
                PD_params[0] * PD_params[2] * PD_params[1]/(num_tr+1)
            ] * (num_tr+1) + [PD_params[0]]
        elif model == "Friberg":
            # PD_params = ["R_0", "MTT", "gamma", ...]
            if len(PD_params) != 4+int(E_max):
                raise ValueError(
                    'Expected ' + str(4+int(E_max)) +
                    ' parameters, but recieved ' +
                    str(len(PD_params)) + ' Parameters.'
                )
            self.num_tr = num_tr
            k_tr = (num_tr+1)/PD_params[1]
            self.PD_params = {
                "R_0": PD_params[0],
                "k_prol": k_tr,
                "k_out": k_tr,
                "k_tr": k_tr,
                "gamma": PD_params[2]
            }
            self.y_0 = [PD_params[0]]*(num_tr+2)
        else:
            raise ValueError(
                'model must be one of "Danyeka", "Zamboni" or "Friberg".'
            )
        self.model = model

        if E_max:
            self.PD_params["s"] = 0
            self.PD_params["EC_50"] = PD_params[-1]
            self.PD_params["E_max"] = PD_params[-2]
        else:
            self.PD_params["s"] = PD_params[-1]
            self.PD_params["EC_50"] = 1
            self.PD_params["E_max"] = 0

    def drug_conc(self, t):
        conc = self.drug_amt(t)[0]/self.PK_params[0]
        return conc

    def E_Drug_lin(self, conc, s):
        return s*conc

    def E_Drug_Emax(self, conc, E_max, EC_50):
        return (E_max*conc)/(EC_50+conc)

    def prol_zero_order(self):
        production = (
            self.PD_params["k_in"] *
            (1-self.E_Drug_lin(self.current["Conc"], self.PD_params["s"])) *
            (1-self.E_Drug_Emax(
                self.current["Conc"],
                self.PD_params["E_max"],
                self.PD_params["EC_50"]
            ))
        )
        return production

    def prol_first_order(self):
        E_drug1 = self.E_Drug_lin(
            self.current["Conc"], self.PD_params["s"]
        )
        E_drug2 = self.E_Drug_Emax(
            self.current["Conc"],
            self.PD_params["E_max"],
            self.PD_params["EC_50"]
        )
        E_drug = (1-E_drug1)*(1-E_drug2)
        production = self.PD_params["k_prol"] * self.current["P"] * E_drug
        return production

    def feedback(self):
        feedback = np.power(
                (self.PD_params["R_0"]/self.current["R"]),
                self.PD_params["gamma"]
            )
        return feedback

    def transfer(self):
        return self.PD_params["k_tr"]*np.concatenate(
            (np.asarray([self.current["P"]]), self.current["T"])
        )

    def elimination(self):
        return self.PD_params["k_out"]*self.current["R"]

    def PD_dydt(self, t, y):
        self.current = {"Conc": self.drug_conc(t)}
        if self.model == "Danyeka":
            self.current["R"] = y
            dy_dt = (
                self.prol_zero_order() -
                self.elimination()
            )
        else:
            self.current.update({"P": y[0], "T": y[1:-1], "R": y[-1]})
            trans = self.transfer()
            if self.model == "Zamboni":
                dP_dt = self.prol_zero_order() - trans[0]
            elif self.model == "Friberg":
                dP_dt = (
                    self.prol_first_order() * self.feedback() - trans[0]
                )
            dT_dt = trans[0:-1] - trans[1:]
            dR_dt = trans[-1] - self.elimination()
            dy_dt = np.concatenate(
                (np.asarray([dP_dt]), dT_dt, np.asarray([dR_dt]))
            )
        return dy_dt
    

    def PD_result(self, times, nadir=False):
        t_before_0 = np.count_nonzero(times < 0)

        results_1 = np.asarray([self.y_0]*t_before_0)
        if nadir:
            def find_nadir(t, y): return self.PD_dydt(t, y)[-1]
            def find_baseline(t,y): 
                if t<0.1:
                    return -0.1
                else:
                    return y[-1]-self.PD_params['R_0']
            find_baseline.terminal = True
            find_baseline.direction = +1
            find_nadir.direction = +1
            results_2 = integrate.solve_ivp(
                self.PD_dydt,
                (0, np.max(times)),
                self.y_0,
#                 t_eval=np.concatenate(
#                     (np.asarray([0]), times[t_before_0:])
#                 ),
                rtol = 1e-3*self.tolerance, 
                atol = 1e-6*self.tolerance,
                dense_output = True,
                events = [find_nadir, find_baseline]
            )
            def AUB(t): return self.PD_params['R_0'] - results_2.sol(t)[-1]
            nadir_value = results_2.y_events[0]
            nadir_value = nadir_value[0,-1]
            AUB_value = integrate.quad(AUB, 0, (results_2.t_events[1])[0])
            
            return nadir_value, AUB_value[0]
        else:
            results_2 = integrate.solve_ivp(
                self.PD_dydt,
                (0, np.max(times)),
                self.y_0,
                t_eval=np.concatenate(
                    (np.asarray([0]), times[t_before_0:])
                ),
                rtol = 1e-3*self.tolerance, 
                atol = 1e-6*self.tolerance 
            )
        if results_2.success:
            results_conc = np.concatenate(
                (results_1, np.transpose(results_2.y[:, 1:]))
            )
        else:
            extra_len = len(times[t_before_0:]) - len(results_2.y[0, 1:])
            extend_results = (
                np.asarray([[1e8]*len(self.y_0)]*extra_len) *
                np.sign(results_2.y[:, -1])
            )
            results_conc = np.concatenate(
                (results_1, np.transpose(results_2.y[:, 1:]), extend_results)
            )
        return results_conc


# if __name__ == '__main__':
#     # Params:

    # R_0 = 5.45
    # MTT = 135
    # gamma = 0.174
    # s = 0.126

    # n = 3
    # k_tr = (n+1)/MTT
    # k_prol = k_tr
    # k_circ = k_tr
    # dose = 2

    # PK_params = np.load('simulated_parameters_actual_dose'+str(dose)+'.npy')
    # print("PK Parameters:", PK_params)
    # PD_params_1 = [k_tr, k_prol, k_circ, R_0, gamma, s]
    # print("PD Parameters:", PD_params_1)
    # PD_params_2 = [R_0, MTT, gamma, s]
    # print("PD Parameters:", PD_params_2)

    # times = np.linspace(-0.1, 1440, 1441)
    # y_0 = [R_0]*5
    # results = integrate.odeint(
    #     PD_friberg_dydt,
    #     y_0,
    #     times,
    #     args=(PD_params_1, PK_params, 2, dose)
    # )
    # plt.plot(times/24, results[:, 4])

    # times = np.linspace(-48, 1440, 1489)
    # parameters = np.concatenate((PK_params, np.asarray(PD_params_2)))
    # results = PD_friberg_result(dose, 2, parameters, times)
    # plt.plot(times/24, results[:])
    # plt.show()


class PintsPDFribergLinE(pints.ForwardModel):
    def __init__(
        self,
        PK_params,
        data=None,
        patient_info=None,
        num_comp=2,
        dose_comp=False,
        fix_param=None
    ):
        super(PintsPDFribergLinE, self).__init__()

        if data is None:
            self.data = pd.DataFrame({'ID': [0]})
            self.single_patient = True
            self._start_time = 0
        else:
            self.data = data
            self._start_time = data.groupby('ID')['TIME'].min()
            time_data = data.merge(
                self._start_time,
                how='left',
                on='ID',
                suffixes=(None, '_start')
            )
            self.pseudotime = (
                (time_data['ID']) *
                np.power(10, np.ceil(np.log10(time_data['TIME'].max())))
                + time_data['TIME']-time_data['TIME_start']
            )
            self.single_patient = False

        if patient_info is None:
            dose_amt = self.data.drop_duplicates(['ID', 'DOSE'])
            dose_amt = dose_amt[['ID', 'DOSE']]
            self.dose = dict(dose_amt.values)
        elif isinstance(patient_info, pd.DataFrame):
            dose_amt = patient_info.groupby(['DOSE'])['AMT'].mean()
            dose_amt = patient_info[['ID', 'DOSE']].join(
                dose_amt, on='DOSE', how='left'
            )
            self.dose = dict(dose_amt[['ID', 'AMT']].values)
        else:
            dose_amt = patient_info['AMT']
            ids = self.data.ID.unique()
            self.dose_amt = dict(zip(ids, [dose_amt]*len(ids)))
            if 'start_time' in patient_info:
                self._start_time = patient_info['start_time']
        self.dose_comp = dose_comp

        if fix_param is None:
            self._fix_param = [None]*4
        else:
            self._fix_param = fix_param

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
            self._times = [-48, 1000]
        self.dose_model = {}
        self.PK_params = PK_params
        self.num_comp = num_comp
        for dose in self._unique_doses:
            self.dose_model[dose] = PD_model()
            self.dose_model[dose].create_PK(
                dose, PK_params, np.max(self._times), num_comp=num_comp
            )

    def n_parameters(self):
        return (self._fix_param.count(None))

    def simulate(self, parameter, times):
        if self.single_patient:
            times = times + self._start_time
            return self.single_simulate(parameter, times, self.dose_amt[0])
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
        PD_params = parameter
        max_times = np.max(times)
        max_self = np.max(self._times)
        for i, param_value in enumerate(self._fix_param):
            if param_value is not None:
                PD_params = np.insert(PD_params, i, param_value)
        if dose_amt in self._unique_doses and max_times <= max_self:
            self.dose_model[dose_amt].set_PD_model("Friberg", PD_params)
            curve = self.dose_model[dose_amt].PD_result(times)[:, -1]
        else:
            self._unique_doses.add(dose_amt)
            self.dose_model[dose_amt] = PD_model()
            self.dose_model[dose_amt].create_PK(
                dose_amt,
                self.PK_params,
                max(max_times, max_self),
                num_comp=self.num_comp
            )
            self.dose_model[dose_amt].set_PD_model("Friberg", PD_params)
            curve = self.dose_model[dose_amt].PD_result(times)[:, -1]
        return curve
    
    def single_nadir(self, parameter, dose_amt, max_time=None):
        for i, param_value in enumerate(self._fix_param):
            if param_value is not None:
                parameter = np.insert(parameter, i, param_value)
        max_self = np.max(self._times)
        if max_time==None:
            max_time = max_self
        if not (dose_amt in self._unique_doses and max_time <= max_self):
            self.dose_model[dose_amt] = PD_model()
            self.dose_model[dose_amt].create_PK(
                dose_amt,
                self.PK_params,
                max(max_time, max_self),
                num_comp=self.num_comp
            )
            self._unique_doses.add(dose_amt)
        self.dose_model[dose_amt].set_PD_model("Friberg", parameter)
        nadir, AUB = self.dose_model[dose_amt].PD_result(np.asarray([max_time]), nadir=True)
        return nadir, AUB
    
    def nadir(self, parameter, max_dose, max_time=None, num_eval=100):
        doses = np.linspace(0, max_dose, num_eval)
        nadir = np.asarray([doses, [0]*num_eval])
        AUB = np.asarray([doses, [0]*num_eval])
        for i, dose_amt in enumerate(doses):
            if dose_amt==0:
                nadir[1, i], AUB[1, i] = parameter[0], 0
            else:
                result = self.single_nadir(parameter, dose_amt, max_time=max_time)
                nadir[1, i] = result[0]
                AUB[1, i] = result[1]
        return nadir, AUB
        
    