import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import pints
from Code.PK_model import PK_result
import myokit
import chi
import copy


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


class ChiMyelotoxicityPKPD(chi.MechanisticModel):
    def __init__(
        self, PK_model, drug_variable, PD_model="Friberg", E_max=False, tr_comps=3
    ):
        super(ChiMyelotoxicityPKPD, self).__init__()

        self._PK_model = PK_model
        self.num_comp = PK_model.num_comp
        self._PD_model_name = PD_model
        self._E_max = E_max
        self._drug_variable = drug_variable
        if PD_model in ["Friberg"]:
            self._n_tr_comps = tr_comps
        else:
            self._n_tr_comps = 0

        # Create the model
        myokit.Unit.register(
            'hour',
            myokit.Unit([0, 0, 1, 0, 0, 0, 0], np.log10(60**2)),
            quantifiable=True, output=True
        )
        self._PD_model = self.make_myokit_model()
        self.set_units(model="PD")

        self._model = self._PD_model.clone()
        PK_comps = [
            comp for comp in self._PK_model._model.components()
            if comp.name() != 'time'
        ]
        PK_names = np.char.array([comp.name() for comp in PK_comps])
        PK_names = 'PK_' + PK_names
        self._model.import_component(
            PK_comps,
            var_map={drug_variable: 'drug.drug_concentration'},
            new_name=PK_names,
            convert_units=True
        )
        drug_variable = 'PK_' + drug_variable
        E_drug = self._model['E_drug']['E_drug']
        if self._E_max:
            E_drug.set_rhs("(E_max*"+drug_variable+")/("+drug_variable+"+EC_50)")
        else:
            E_drug.set_rhs("S*"+drug_variable)

        # Set default number and names of states, parameters and outputs.
        self._set_number_and_names()

        # Get time unit
        self._time_unit = self._model.time().unit()

        # Create simulator without sensitivities
        # (intentionally public property)
        self._simulator = myokit.Simulation(self._model)
        self._has_sensitivities = False

        # Set default dose administration
        self._administration = None
        self._dosing_regimen = None

        # Safe vanilla model
        self._vanilla_model = self._model.clone()
        self.set_outputs(['circulating.R'])

    def make_myokit_model(self):
        self.name = (
            self._PD_model_name +
            " Model of Myelotoxicity with " +
            self._E_max * "E_max " +
            (not self._E_max) * "Linear "+
            "Drug effect"
        )
        model = myokit.Model(name=self.name)

        drug = model.add_component("drug")
        drug.add_variable("drug_concentration").set_rhs(0)

        E_drug_comp = model.add_component("E_drug")
        E_drug = E_drug_comp.add_variable("E_drug")
        if self._E_max:
            E_drug_comp.add_variable("E_max").set_rhs(1)
            E_drug_comp.add_variable("EC_50").set_rhs(1)
            E_drug.set_rhs("(E_max*drug.drug_concentration)/(drug.drug_concentration+EC_50)")
        else:
            E_drug_comp.add_variable("S").set_rhs(0.5)
            E_drug.set_rhs("S*drug.drug_concentration")
        
        proliferating = model.add_component("proliferating")
        P = proliferating.add_variable("P")
        K_in = proliferating.add_variable("K_in")
        eq_P = "K_in*(1-E_drug.E_drug)"
        if self._PD_model_name == "Friberg":
            proliferating.add_variable("gamma").set_rhs(0.44)
            eq_P += "*((circulating.R_0/circulating.R)^gamma)"
            eq_P += "*P"
        if self._PD_model_name != "Dayneka":
            transit = model.add_component("transit")
            transit.add_variable("MTT").set_rhs(85.26)
            transit.add_variable("K_tr").set_rhs(str(self._n_tr_comps+1)+"/MTT")
            eq_P += "-transit.K_tr*P"
            P.promote(983.1)

        circulating = model.add_component("circulating")
        R = circulating.add_variable("R")
        circulating.add_variable("R_0").set_rhs(983.1)
        K_out = circulating.add_variable("K_out")
        if self._PD_model_name == "Friberg":
            K_in.set_rhs("transit.K_tr")
            K_out.set_rhs("transit.K_tr")
        else:
            K_in.set_rhs("circulating.K_out*circulating.R_0")
            K_out.set_rhs(0.03)
        R.promote(983.1)
        eq_R = "- K_out*R"
        if self._PD_model_name == "Dayneka":
            eq_R += "+ proliferating.P "
        elif self._n_tr_comps == 0:
            eq_R += "+ transit.K_tr*proliferating.P"
        else:
            # transit = []
            for i in range(0, self._n_tr_comps):
                T = transit.add_variable("T"+str(i+1))
                T.promote(983.1)
                # transit.append()
                if i==0:
                    eq_t = "K_tr*proliferating.P"
                else:
                    eq_t = "K_tr*T"+str(i)
                eq_t += "-K_tr*T"+str(i+1)
                T.set_rhs(eq_t)
            eq_R += "+ transit.K_tr*transit.T"+str(self._n_tr_comps)

        P.set_rhs(eq_P)
        R.set_rhs(eq_R)

        time = model.add_component("time").add_variable("t")
        time.set_rhs(0)
        time.set_binding("time")

        model.validate(remove_unused_variables=False)
        return model

    def set_units(self, model="PKPD", units={'time': 'hour', 'volume': 'uL', 'amount': 'kg/g'}):
        if model == "PD":
            model = self._PD_model
        elif model == "PKPD":
            model = self._model
        elif model == "PK":
            self._PK_model.set_units(self, units=units)
            return

        time_unit = myokit.parse_unit(units['time'])
        volume_unit = myokit.parse_unit(units['volume'])
        amount_unit = myokit.parse_unit(units['amount'])

        drug_conc_unit = self._PK_model._model.get(self._drug_variable).unit()
        model.time().set_unit(unit=time_unit)
        if self._E_max:
            model['E_drug']['E_max'].set_unit(unit=1)
            model['E_drug']['EC_50'].set_unit(unit=drug_conc_unit)
        else:
            model['E_drug']['S'].set_unit(unit=1/drug_conc_unit)
        model['drug']['drug_concentration'].set_unit(unit=drug_conc_unit)
        for comp in model.components():
            if not (comp.name() == 'time' or comp.name() == 'drug'):
                for var in comp.variables(state=True):
                    var.set_unit(unit=amount_unit/volume_unit)
                for var in comp.variables(state=False):
                    if var.name() == 'R_0':
                        var.set_unit(unit=amount_unit/volume_unit)
                    if var.name() == 'MTT':
                        var.set_unit(unit=time_unit)
                    if var.name()[0] == 'K':
                        var.set_unit(unit=1/time_unit)

        model.validate(remove_unused_variables=False)
        model.check_units()

    def _set_const(self, parameters):
        """
        Sets values of constant model parameters.
        """
        for id_var, var in enumerate(self._const_names):
            self._simulator.set_constant(var, float(parameters[id_var]))

    def _set_state(self, parameters):
        """
        Sets initial values of states.
        """
        parameters = np.array(parameters)
        parameters = parameters[self._original_order]
        self._simulator.set_state(parameters)

    def _set_number_and_names(self):
        """
        Sets the number of states, parameters and outputs, as well as their
        names. If the model is ``None`` the self._model is taken.
        """
        # Get the number of states and parameters
        self._n_states = self._model.count_states()
        n_const = self._model.count_variables(const=True)

        # Get constant variable names and state names
        names = [var.qname() for var in self._model.states()]
        self._state_names = sorted(names)

        const_names = []
        for var in self._model.variables(const=True):
            # Sometimes constants are derived from parameters
            if not var.is_literal():
                n_const -= 1
                continue
            const_names.append(var.qname())
        self._const_names = sorted(const_names)
        self._n_parameters = n_const

        # Remember original order of state names for simulation
        order_after_sort = np.argsort(names)
        self._original_order = np.argsort(order_after_sort)

        # Set default parameter names
        self._parameter_names = self._const_names

        # Temporarily set outputs to default outputs
        self._output_names = self._state_names
        self._n_outputs = self._n_states

        # Create references of displayed parameter and output names to
        # orginal myokit names (defaults to identity map)
        # (Key: myokit name, value: displayed name)
        self._parameter_name_map = dict(
            zip(self._parameter_names, self._parameter_names))
        self._output_name_map = dict(
            zip(self._output_names, self._output_names))

    def _add_dose_compartment(self, model, drug_amount):
        """
        Adds a dose compartment to the model with a linear absorption rate to
        the connected compartment.
        """
        # Add a dose compartment to the model
        dose_comp = model.add_component_allow_renaming('dose')

        # Create a state variable for the drug amount in the dose compartment
        dose_drug_amount = dose_comp.add_variable('drug_amount')
        dose_drug_amount.set_rhs(0)
        dose_drug_amount.set_unit(drug_amount.unit())
        dose_drug_amount.promote()

        # Create an absorption rate variable
        absorption_rate = dose_comp.add_variable('absorption_rate')
        absorption_rate.set_rhs(1)
        absorption_rate.set_unit(1 / self.time_unit())

        # Add outflow expression to dose compartment
        dose_drug_amount.set_rhs(
            myokit.Multiply(
                myokit.PrefixMinus(myokit.Name(absorption_rate)),
                myokit.Name(dose_drug_amount)
                )
            )

        # Add inflow expression to connected compartment
        rhs = drug_amount.rhs()
        drug_amount.set_rhs(
            myokit.Plus(
                rhs,
                myokit.Multiply(
                    myokit.Name(absorption_rate),
                    myokit.Name(dose_drug_amount)
                )
            )
        )

        # Update number of parameters and states, as well as their names
        # (This overwrites current outputs, so we have to set them again)
        self._model = model
        original_outputs = self._output_names
        self._set_number_and_names()
        self.set_outputs(original_outputs)

        return model, dose_drug_amount

    def _add_dose_rate(self, compartment, drug_amount):
        """
        Adds a dose rate variable to the state variable, which is bound to the
        dosing regimen.
        """
        # Register a dose rate variable to the compartment and bind it to
        # pace, i.e. tell myokit that its value is set by the dosing regimen/
        # myokit.Protocol
        dose_rate = compartment.add_variable_allow_renaming(
            str('dose_rate'))
        dose_rate.set_binding('pace')

        # Set initial value to 0 and unit to unit of drug amount over unit of
        # time
        dose_rate.set_rhs(0)
        dose_rate.set_unit(drug_amount.unit() / self.time_unit())

        # Add the dose rate to the rhs of the drug amount variable
        rhs = drug_amount.rhs()
        drug_amount.set_rhs(
            myokit.Plus(
                rhs,
                myokit.Name(dose_rate)
            )
        )

    def administration(self):
        """
        Returns the mode of administration in form of a dictionary.

        The dictionary has the keys 'compartment' and 'direct'. The former
        provides information about which compartment is dosed, and the latter
        whether the dose is administered directly ot indirectly to the
        compartment.
        """
        return self._administration

    def copy(self):
        """
        Returns a deep copy of the mechanistic model.

        .. note::
            Copying the model resets the sensitivity settings.
        """
        # Copy model manually and get protocol
        m = self._model.clone()
        s = self._simulator
        myokit_model = m.clone()
        self._model = None
        self._simulator = None

        # Copy the mechanistic model
        model = copy.deepcopy(self)

        # Replace myokit model by safe copy and create simulator
        self._model = m
        self._simulator = s
        model._model = myokit_model
        model._simulator = myokit.Simulation(myokit_model)
        model._simulator.set_protocol(model.dosing_regimen())

        return model

    def dosing_regimen(self):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`myokit.Protocol`. If the protocol has not been set, ``None`` is
        returned.
        """
        return self._dosing_regimen

    def enable_sensitivities(self, enabled, parameter_names=None):
        """
        Enables the computation of the model output sensitivities to the model
        parameters if set to ``True``.

        The sensitivities are computed using the forward sensitivities method,
        where an ODE for each sensitivity is derived. The sensitivities are
        returned together with the solution to the orginal system of ODEs when
        simulating the mechanistic model :meth:`simulate`.

        The optional parameter names argument can be used to set which
        sensitivities are computed. By default the sensitivities to all
        parameters are computed.

        :param enabled: A boolean flag which enables (``True``) / disables
            (``False``) the computation of sensitivities.
        :type enabled: bool
        :param parameter_names: A list of parameter names of the model. If
            ``None`` sensitivities for all parameters are computed.
        :type parameter_names: list[str], optional
        """
        enabled = bool(enabled)

        # Check whether myokit.Simulation needs to be updated
        new_sim = False
        if enabled or ((not enabled) and self._has_sensitivities):
            new_sim = True
        
        if not enabled:
            if self._has_sensitivities:
                # Disable sensitivities
                sim = myokit.Simulation(self._model)
                self._simulator = sim
                self._has_sensitivities = False

                return None

            # Sensitivities are already disabled
            return None

        # Get parameters whose output sensitivities are computed
        parameters = []
        for param_id, param in enumerate(self._parameter_names):
            # Other parameters can be appended without modification
            parameters.append(param)

        if parameter_names is not None:
            # Get myokit names for input parameter names
            container = []
            for index, public_name in enumerate(
                    self._parameter_name_map.values()):
                if public_name in parameter_names:
                    container.append(parameters[index])

            parameters = container

        if not parameters:
            raise ValueError(
                'None of the parameters could be identified. The valid '
                'parameter names are <' + str(self._parameter_names) + '>.')

        # Create simulator
        sensitivities = (self._output_names, parameters)
        sim = myokit.Simulation(
            self._model, protocol=None, sensitivities=sensitivities)

        # Update simulator and sensitivity state
        self._simulator = sim
        self._has_sensitivities = True

        # Update dosing regimen if sensitivity has resulted in new
        # myokit.Simulation instance
        if new_sim:
            self._simulator.set_protocol(self._dosing_regimen)

    def has_sensitivities(self):
        """
        Returns a boolean indicating whether sensitivities have been enabled.
        """
        return self._has_sensitivities

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._n_parameters
    
    def outputs(self):
        """
        Returns the output names of the model.
        """
        # Get user specified output names
        output_names = [
            self._output_name_map[name] for name in self._output_names]
        return output_names

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        # Get user specified parameter names
        parameter_names = [
            self._parameter_name_map[name] for name in self._parameter_names]

        return parameter_names

    def set_administration(
            self, compartment, amount_var='drug_amount', direct=True):
        r"""
        Sets the route of administration of the compound.

        The compound is administered to the selected compartment either
        directly or indirectly. If it is administered directly, a dose rate
        variable is added to the drug amount's rate of change expression

        .. math ::

            \frac{\text{d}A}{\text{d}t} = \text{RHS} + r_d,

        where :math:`A` is the drug amount in the selected compartment, RHS is
        the rate of change of :math:`A` prior to adding the dose rate, and
        :math:`r_d` is the dose rate.

        The dose rate can be set by :meth:`set_dosing_regimen`.

        If the route of administration is indirect, a dosing compartment
        is added to the model, which is connected to the selected compartment.
        The dose rate variable is then added to the rate of change expression
        of the dose amount variable in the dosing compartment. The drug amount
        in the dosing compartment flows at a linear absorption rate into the
        selected compartment

        .. math ::

            \frac{\text{d}A_d}{\text{d}t} = -k_aA_d + r_d \\
            \frac{\text{d}A}{\text{d}t} = \text{RHS} + k_aA_d,

        where :math:`A_d` is the amount of drug in the dose compartment and
        :math:`k_a` is the absorption rate.

        Setting an indirect administration route changes the number of
        parameters of the model, because an initial dose compartment drug
        amount and a absorption rate parameter are added.

        .. note:
            Setting the route of administration will reset the sensitivity
            settings.

        :param compartment: Compartment to which doses are either directly or
            indirectly administered.
        :type compartment: str
        :param amount_var: Drug amount variable in the compartment. By default
            the drug amount variable is assumed to be 'drug_amount'.
        :type amount_var: str, optional
        :param direct: A boolean flag that indicates whether the dose is
            administered directly or indirectly to the compartment.
        :type direct: bool, optional
        """
        # Check inputs
        model = self._vanilla_model.clone()
        if not model.has_component(compartment):
            raise ValueError(
                'The model does not have a compartment named <'
                + str(compartment) + '>.')
        comp = model.get(compartment, class_filter=myokit.Component)

        if not comp.has_variable(amount_var):
            raise ValueError(
                'The drug amount variable <' + str(amount_var) + '> could not '
                'be found in the compartment.')

        drug_amount = comp.get(amount_var)
        if not drug_amount.is_state():
            raise ValueError(
                'The variable <' + str(drug_amount) + '> is not a state '
                'variable, and can therefore not be dosed.')

        # If administration is indirect, add a dosing compartment and update
        # the drug amount variable to the one in the dosing compartment
        if not direct:
            model, drug_amount = self._add_dose_compartment(model, drug_amount)
            comp = model.get(compartment, class_filter=myokit.Component)

        # Add dose rate variable to the right hand side of the drug amount
        self._add_dose_rate(comp, drug_amount)

        # Update model and simulator
        # (otherwise simulator won't know about pace bound variable)
        self._model = model
        self._simulator = myokit.Simulation(model)
        self._has_sensitivities = False

        # Remember type of administration
        self._administration = dict(
            {'compartment': compartment, 'direct': direct})

    def set_dosing_regimen(
            self, dose, start=0, duration=0.01, period=None, num=None):
        """
        Sets the dosing regimen with which the compound is administered.

        The route of administration can be set with :meth:`set_administration`.
        However, the type of administration, e.g. bolus injection or infusion,
        may be controlled with the duration input.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model an infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the dose is administered once. To apply multiple doses
        provide a dose administration period.

        :param dose: The amount of the compound that is injected at each
            administration, or a myokit.Protocol instance that defines the
            dosing regimen.
        :type dose: float or myokit.Protocol
        :param start: Start time of the treatment. By default the
            administration starts at t=0.
        :type start: float, optional
        :param duration: Duration of dose administration. By default the
            duration is set to 0.01 of the time unit (bolus).
        :type duration: float, optional
        :param period: Periodicity at which doses are administered. If ``None``
            the dose is administered only once.
        :type period: float, optional
        :param num: Number of administered doses. If ``None`` and the
            periodicity of the administration is not ``None``, doses are
            administered indefinitely.
        :type num: int, optional
        """
        if self._administration is None:
            raise ValueError(
                'The route of administration of the dose has not been set.')

        if num is None:
            # Myokits default is zero, i.e. infinitely many doses
            num = 0

        if period is None:
            # If period is not provided, we administer a single dose
            # Myokits defaults are 0s for that.
            period = 0
            num = 0

        if isinstance(dose, myokit.Protocol):
            self._simulator.set_protocol(dose)
            self._dosing_regimen = dose
            return None

        # Translate dose to dose rate
        dose_rate = dose / duration

        # Set dosing regimen
        dosing_regimen = myokit.pacing.blocktrain(
            period=period, duration=duration, offset=start, level=dose_rate,
            limit=num)
        self._simulator.set_protocol(dosing_regimen)
        self._dosing_regimen = dosing_regimen

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.

        The outputs can be set to any quantifiable variable name of the
        :class:`myokit.Model`, e.g. `compartment.variable`.

        .. note::
            Setting outputs resets the sensitivity settings (by default
            sensitivities are disabled.)

        :param outputs:
            A list of output names.
        :type outputs: list[str]
        """
        outputs = list(outputs)

        # Translate public names to myokit names, if set previously
        for myokit_name, public_name in self._output_name_map.items():
            if public_name in outputs:
                # Replace public name by myokit name
                index = outputs.index(public_name)
                outputs[index] = myokit_name

        # Check that outputs are valid
        for output in outputs:
            try:
                var = self._simulator._model.get(output)
                if not (var.is_state() or var.is_intermediary()):
                    raise ValueError(
                        'Outputs have to be state or intermediary variables.')
            except KeyError:
                raise KeyError(
                    'The variable <' + str(output) + '> does not exist in the '
                    'model.')

        # Remember outputs
        self._output_names = outputs
        self._n_outputs = len(outputs)

        # Create an updated output name map
        output_name_map = {}
        for myokit_name in self._output_names:
            try:
                output_name_map[myokit_name] = self._output_name_map[
                    myokit_name]
            except KeyError:
                # The output did not exist before, so create an identity map
                output_name_map[myokit_name] = myokit_name
        self._output_name_map = output_name_map

        # Disable sensitivities
        self.enable_sensitivities(False)

    def set_output_names(self, names):
        """
        Assigns names to the model outputs. By default the
        :class:`myokit.Model` names are assigned to the outputs.

        :param names: A dictionary that maps the current output names to new
            names.
        :type names: dict[str, str]
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current output names'
                'as keys and the new output names as values.')

        # Check that new output names are unique
        new_names = list(names.values())
        n_unique_new_names = len(set(names.values()))
        if len(new_names) != n_unique_new_names:
            raise ValueError(
                'The new output names have to be unique.')

        # Check that new output names do not exist already
        for new_name in new_names:
            if new_name in list(self._output_name_map.values()):
                raise ValueError(
                    'The output names cannot coincide with existing '
                    'output names. One output is already called '
                    '<' + str(new_name) + '>.')

        # Replace currently displayed names by new names
        for myokit_name in self._output_names:
            old_name = self._output_name_map[myokit_name]
            try:
                new_name = names[old_name]
                self._output_name_map[myokit_name] = str(new_name)
            except KeyError:
                # KeyError indicates that the current output is not being
                # renamed.
                pass

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        :param names: A dictionary that maps the current parameter names to new
            names.
        :type names: dict[str, str]
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current parameter names'
                'as keys and the new parameter names as values.')

        # Check that new parameter names are unique
        new_names = list(names.values())
        n_unique_new_names = len(set(names.values()))
        if len(new_names) != n_unique_new_names:
            raise ValueError(
                'The new parameter names have to be unique.')

        # Check that new parameter names do not exist already
        for new_name in new_names:
            if new_name in list(self._parameter_name_map.values()):
                raise ValueError(
                    'The parameter names cannot coincide with existing '
                    'parameter names. One parameter is already called '
                    '<' + str(new_name) + '>.')

        # Replace currently displayed names by new names
        for myokit_name in self._parameter_names:
            old_name = self._parameter_name_map[myokit_name]
            try:
                new_name = names[old_name]
                self._parameter_name_map[myokit_name] = str(new_name)
            except KeyError:
                # KeyError indicates that the current parameter is not being
                # renamed.
                pass

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs (and optionally
        the sensitivites) for the specified parameters and times.

        The model outputs are returned as a 2 dimensional NumPy array of shape
        ``(n_outputs, n_times)``. If sensitivities are enabled, a tuple is
        returned with the NumPy array of the model outputs and a NumPy array of
        the sensitivities of shape ``(n_times, n_outputs, n_parameters)``.

        :param parameters: An array-like object with values for the model
            parameters.
        :type parameters: list, numpy.ndarray
        :param times: An array-like object with time points at which the output
            values are returned.
        :type times: list, numpy.ndarray

        :rtype: np.ndarray of shape (n_outputs, n_times) or
            (n_times, n_outputs, n_parameters)
        """
        # Reset simulation
        self._simulator.reset()

        # Set initial conditions
        y_0 = [parameters[5]]*(self._n_states-self._PK_model.num_comp)
        for i in range(self._PK_model.num_comp):
            y_0.insert(-(self._n_tr_comps+2), 0)
        self._set_state(y_0)

        # Set constant model parameters
        self._set_const(parameters)

        # Simulate
        if not self._has_sensitivities:
            output = self._simulator.run(
                times[-1] + 1, log=self._output_names, log_times=times)
            output = np.array([output[name] for name in self._output_names])

            return output

        output, sensitivities = self._simulator.run(
            times[-1] + 1, log=self._output_names, log_times=times)
        output = np.array([output[name] for name in self._output_names])
        output
        sensitivities = np.array(sensitivities)

        return output, sensitivities

    def supports_dosing(self):
        """
        Returns a boolean whether dose administration with
        :meth:`PKPDModel.set_dosing_regimen` is supported by the model.
        """
        return True

    def time_unit(self):
        """
        Returns the model's unit of time.
        """
        return self._time_unit