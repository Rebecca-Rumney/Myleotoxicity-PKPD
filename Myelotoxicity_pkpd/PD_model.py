import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pints
from Myelotoxicity_pkpd.PK_model import solve_2_comp


def prol(current_state, PD_Params):
    dProl_dt = (
        PD_Params["k_prol"] *
        current_state["Prol"] *
        (1-E_Drug_lin(current_state["Conc"], PD_Params["slope"])) *
        np.power(
            (PD_Params["Circ_0"]/current_state["Circ"]),
            PD_Params["gamma"]
        ) -
        PD_Params["k_tr"]*current_state["Prol"]
    )
    return dProl_dt


def prol_zero_order(current_state, PD_Params):
    production = (
        PD_Params["k_in"] *
        (1-E_Drug_lin(current_state["Conc"], PD_Params["slope"]))*
        (1-E_Drug_Emax(current_state["Conc"], PD_Params["E_max"], PD_Params["EC_50"]))
    )
    return production


def prol_first_order(current_state, PD_Params):
    production = (
        PD_Params["k_prol"] * current_state["Prol"] *
        (1-E_Drug_lin(current_state["Conc"], PD_Params["slope"]))*
        (1-E_Drug_Emax(current_state["Conc"], PD_Params["E_max"], PD_Params["EC_50"]))
    )
    return production

def feedback(current_state, PD_Params):
    feedback = np.power(
            (PD_Params["Circ_0"]/current_state["Circ"]),
            PD_Params["gamma"]
        )
    return feedback

def transit_dydt(current_state, PD_Params):
    dT1_dt = (
        PD_Params["k_tr"]*current_state["Prol"] -
        PD_Params["k_tr"]*current_state["T1"]
    )
    dT2_dt = (
        PD_Params["k_tr"]*current_state["T1"] -
        PD_Params["k_tr"]*current_state["T2"]
    )
    dT3_dt = (
        PD_Params["k_tr"]*current_state["T2"] -
        PD_Params["k_tr"]*current_state["T3"]
    )
    return (dT1_dt, dT2_dt, dT3_dt)

def transfer(current_state, PD_Params):
    return PD_Params["k_tr"]*np.concatenate((np.asarray([current_state["Prol"]]), current_state["T"]))

def circulation_dydt(current_state, PD_Params):
    dCirc_dt = (
        PD_Params["k_tr"]*current_state["T3"] -
        PD_Params["k_circ"]*current_state["Circ"]
    )
    return dCirc_dt

def elimination(current_state, PD_Params):
    return PD_Params["k_out"]*current_state["Circ"]

def PD(y, t, PD_params, PK_params, num_comp, dose):
    variables = ["Prol", "T1", "T2", "T3", "Circ"]
    current_state = dict(zip(variables, y))
    current_state["Conc"] = drug_conc(t, PK_params, num_comp, dose)*1000

    parameter_names = ["k_tr", "k_prol", "k_circ", "Circ_0", "gamma", "slope"]
    PD_params = dict(zip(parameter_names, PD_params))

    dProl_dt = prol(current_state, PD_params)
    dCirc_dt = circulation_dydt(current_state, PD_params)
    dT1_dt, dT2_dt, dT3_dt = transit_dydt(current_state, PD_params)
    return [dProl_dt, dT1_dt, dT2_dt, dT3_dt, dCirc_dt]

def PD_danyeka_dydt(y, t, PD_params, PK_params, num_comp, dose):
    current_state = {"Circ": y}
    current_state["Conc"] = drug_conc(t, PK_params, num_comp, dose)
    
    dCirc_dt = prol_zero_order(current_state, PD_params) - elimination(current_state, PD_params)
    return dCirc_dt

def PD_zamboni_dydt(y, t, PD_params, PK_params, num_comp, dose):
    current_state = {"Prol": y[0], "T":y[1:-1], "Circ":y[-1]}
    current_state["Conc"] = drug_conc(t, PK_params, num_comp, dose)
    
    trans = transfer(current_state, PD_params)
    dProl_dt = prol_zero_order(current_state, PD_params) - trans[0]
    dT_dt = trans[0:-1] - trans[1:]
    dCirc_dt = trans[-1] - elimination(current_state, PD_params)
    
    return np.concatenate((np.asarray([dProl_dt]), dT_dt, np.asarray([dCirc_dt])))


def PD_friberg_dydt(y, t, PD_params, PK_params, num_comp, dose):
    current_state = {"Prol": y[0], "T":y[1:-1], "Circ":y[-1]}
    current_state["Conc"] = drug_conc(t, PK_params, num_comp, dose)
    
    trans = transfer(current_state, PD_params)
    
    dProl_dt = prol_first_order(current_state, PD_params)*feedback(current_state, PD_params) - trans[0]
    dT_dt = trans[0:-1] - trans[1:]
    dCirc_dt = trans[-1] - elimination(current_state, PD_params)
    
    return np.concatenate((np.asarray([dProl_dt]), dT_dt, np.asarray([dCirc_dt])))

def PD_result(dose, num_comp, parameter, times):
    if len(parameter) != 2*num_comp+4:
        raise ValueError(
            'Expected ' + str(2*num_comp+4) + 
            ' parameters for ' + str(num_comp) + 
            ' compartments, but recieved ' +
            str(len(parameter)) + ' Parameters.'
        )
    PK_params = parameter[:2*num_comp]

    n = 3
    Circ_0 = parameter[2*num_comp]
    MTT = parameter[2*num_comp+1]
    gamma = parameter[2*num_comp+2]
    slope = parameter[2*num_comp+3]

    k_tr = (n+1)/MTT
    k_prol = k_tr
    k_circ = k_tr

    y_0 = [Circ_0]*5
    PD_params = [k_tr, k_prol, k_circ, Circ_0, gamma, slope]
    times_before_dose = np.count_nonzero(times < 0)

    tolerance = 1.49012e-12  # ODEint usually uses 1.49012e-8
    
    results_1, dict_1 = integrate.odeint(
        PD,
        y_0,
        np.concatenate((times[:times_before_dose], np.asarray([0]))),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    results_2, dict_2 = integrate.odeint(
        PD,
        results_1[-1, :],
        np.concatenate((np.asarray([0]), times[times_before_dose:])),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    
    results_conc = np.concatenate((results_1[:-1, 4], results_2[1:, 4]))
    hu = np.concatenate((dict_1['hu'], dict_2['hu']))
    
    return results_conc, hu

def PD_danyeka_result(dose, num_comp, parameter, times, E_max=False):
    if len(parameter) != 2*num_comp+3+int(E_max):
        raise ValueError(
            'Expected ' + str(2*num_comp+3+int(E_max)) + 
            ' parameters for but recieved ' +
            str(len(parameter)) + ' Parameters.'
        )
    PK_params = parameter[:2*num_comp]
    PD_params = parameter[2*num_comp:]
    
    if E_max:
        parameter_names = ["Circ_0", "k_out", "E_max", "EC_50"]
        PD_params = dict(zip(parameter_names, PD_params))
        PD_params["slope"] = 0
    else:
        parameter_names = ["Circ_0", "k_out", "slope"]
        PD_params = dict(zip(parameter_names, PD_params))
        PD_params["E_max"] = 0
        PD_params["EC_50"] = 1
    
    PD_params["k_in"] = PD_params["k_out"]*PD_params["Circ_0"]
    y_0 = PD_params["Circ_0"]
    times_before_dose = np.count_nonzero(times < 0)

    tolerance = 1.49012e-12  # ODEint usually uses 1.49012e-8
    
    results_1, dict_1 = integrate.odeint(
        PD_danyeka_dydt,
        y_0,
        np.concatenate((times[:times_before_dose], np.asarray([0]))),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    results_2, dict_2 = integrate.odeint(
        PD_danyeka_dydt,
        results_1[-1, :],
        np.concatenate((np.asarray([0]), times[times_before_dose:])),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    
    results_conc = np.concatenate((results_1[:-1], results_2[1:]))
    hu = np.concatenate((dict_1['hu'], dict_2['hu']))
    
    return results_conc

def PD_zamboni_result(dose, num_PK_comp, parameter, times, E_max=False, num_transit=1):
    if len(parameter) != 2*num_PK_comp+4+int(E_max):
        raise ValueError(
            'Expected ' + str(2*num_PK_comp+4+int(E_max)) + 
            ' parameters for but recieved ' +
            str(len(parameter)) + ' Parameters.'
        )
    PK_params = parameter[:2*num_PK_comp]
    PD_params = parameter[2*num_PK_comp:]
    
    if E_max:
        parameter_names = ["Circ_0", "MTT", "k_out", "E_max", "EC_50"]
        PD_params = dict(zip(parameter_names, PD_params))
        PD_params["slope"] = 0
    else:
        parameter_names = ["Circ_0", "MTT", "k_out", "slope"]
        PD_params = dict(zip(parameter_names, PD_params))
        PD_params["E_max"] = 0
        PD_params["EC_50"] = 1
    
    
    PD_params["k_tr"] = (num_transit+1)/PD_params["MTT"]
    PD_params["k_in"] = PD_params["Circ_0"]*PD_params["k_out"]
    y_0 = [PD_params["k_in"]/PD_params["k_tr"]]*(num_transit+1)+[PD_params["Circ_0"]]
    times_before_dose = np.count_nonzero(times < 0)

    tolerance = 1.49012e-12  # ODEint usually uses 1.49012e-8
    
    results_1, dict_1 = integrate.odeint(
        PD_zamboni_dydt,
        y_0,
        np.concatenate((times[:times_before_dose], np.asarray([0]))),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    results_2, dict_2 = integrate.odeint(
        PD_zamboni_dydt,
        results_1[-1, :],
        np.concatenate((np.asarray([0]), times[times_before_dose:])),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    
    results_conc = np.concatenate((results_1[:-1], results_2[1:]))
    hu = np.concatenate((dict_1['hu'], dict_2['hu']))
    
    return results_conc

def PD_friberg_result(dose, num_PK_comp, parameter, times, E_max=False, num_transit=3):
    if len(parameter) != 2*num_PK_comp+4+int(E_max):
        raise ValueError(
            'Expected ' + str(2*num_PK_comp+4+int(E_max)) + 
            ' parameters for but recieved ' +
            str(len(parameter)) + ' Parameters.'
        )
    PK_params = parameter[:2*num_PK_comp]
    PD_params = parameter[2*num_PK_comp:]
    
    n=3
    
    if E_max:
        parameter_names = ["Circ_0", "MTT", "gamma", "E_max", "EC_50"]
        PD_params = dict(zip(parameter_names, PD_params))
        PD_params["slope"] = 0
    else:
        parameter_names = ["Circ_0", "MTT", "gamma", "slope"]
        PD_params = dict(zip(parameter_names, PD_params))
        PD_params["E_max"] = 0
        PD_params["EC_50"] = 1
        
    PD_params["k_tr"] = (n+1)/PD_params["MTT"]
    PD_params["k_prol"] = PD_params["k_tr"]
    PD_params["k_out"] = PD_params["k_tr"]

    y_0 = [PD_params["Circ_0"]]*(n+2)
    times_before_dose = np.count_nonzero(times < 0)

    tolerance = 1.49012e-12  # ODEint usually uses 1.49012e-8
    
    results_1, dict_1 = integrate.odeint(
        PD_friberg_dydt,
        y_0,
        np.concatenate((times[:times_before_dose], np.asarray([0]))),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    results_2, dict_2 = integrate.odeint(
        PD_friberg_dydt,
        results_1[-1, :],
        np.concatenate((np.asarray([0]), times[times_before_dose:])),
        args=(PD_params, PK_params, 2, dose),
        atol = tolerance,
        rtol = tolerance,
        full_output = True
    )
    
    results_conc = np.concatenate((results_1[:-1], results_2[1:]))
    hu = np.concatenate((dict_1['hu'], dict_2['hu']))
    
    return results_conc

def drug_conc(t, PK_params, num_comp, dose):
    if num_comp == 2:
        conc = solve_2_comp(dose, PK_params, t)
    return conc


def E_Drug_lin(conc, slope):
    return slope*conc

def E_Drug_Emax(conc, E_max, EC_50):
    return (E_max*conc)/(EC_50+conc)


if __name__ == '__main__':
    # Params:

    Circ_0 = 5.45
    MTT = 135
    gamma = 0.174
    slope = 0.126

    n = 3
    k_tr = (n+1)/MTT
    k_prol = k_tr
    k_circ = k_tr
    dose = 2

    PK_params = np.load('simulated_parameters_actual_dose'+str(dose)+'.npy')
    print("PK Parameters:", PK_params)
    PD_params_1 = [k_tr, k_prol, k_circ, Circ_0, gamma, slope]
    print("PD Parameters:", PD_params_1)
    PD_params_2 = [Circ_0, MTT, gamma, slope]
    print("PD Parameters:", PD_params_2)

    times = np.linspace(-0.1, 1440, 1441)
    y_0 = [Circ_0]*5
    results = integrate.odeint(
        PD,
        y_0,
        times,
        args=(PD_params_1, PK_params, 2, dose)
    )
    plt.plot(times/24, results[:, 4])

    times = np.linspace(-48, 1440, 1489)
    parameters = np.concatenate((PK_params, np.asarray(PD_params_2)))
    results = PD_result(dose, 2, parameters, times)
    plt.plot(times/24, results[:])
    plt.show()


# class PintsPDFriberg(pints.ForwardModel):
#     def __init__(self, PK_params, dose, num_comp=2, start_time=None):
#         super(PintsPDFriberg, self).__init__()
#         self._PK_params = PK_params
#         self._dose = dose
#         self._num_comp = num_comp
#         if start_time is None:
#             self._start_time = 0
#         else:
#             self._start_time = start_time

#     def n_parameters(self):
#         return 4

#     def simulate(self, parameter, times):
#         times = times + self._start_time
#         all_params = np.concatenate((self._PK_params, np.asarray(parameter)))
#         curve = PD_result(self._dose, self._num_comp, all_params, times)
#         return curve


class PintsPDFriberg(pints.ForwardModel):
    def __init__(
        self, PK_params, dose, fix_param=[[],[]], num_comp=2, start_time=None, record_hu = False
    ):
        super(PintsPDFriberg, self).__init__()
        self._PK_params = PK_params
        self._dose = dose
        self._num_comp = num_comp
        self._fix_param = fix_param
        if start_time is None:
            self._start_time = 0
        else:
            self._start_time = start_time
        if record_hu:
            self.hu = []
        else:
            self.hu = None
            

    def n_parameters(self):
        return 4 - len(self._fix_param[0])

    def simulate(self, parameter, times):
        times = times + self._start_time
        PD_params = parameter
        param_pos = self._fix_param[0]
        indexes = np.argsort(self._fix_param[0])
        param_value = self._fix_param[1]

        for i in indexes:
            PD_params = np.insert(PD_params, param_pos[i], param_value[i])
        all_params = np.concatenate((self._PK_params, PD_params))
        curve, sim_hu = PD_result(self._dose, self._num_comp, all_params, times)
        
        if self.hu != None:
            self.hu.append(sim_hu)
        
        return curve
