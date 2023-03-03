import numpy as np
import pandas as pd
from scipy import integrate, linalg, interpolate
from collections.abc import Iterable
import pints
import myokit
import chi
import copy


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
    import matplotlib.pyplot as plt
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


class ChiPKLin(chi.MechanisticModel):
    def __init__(
        self, num_comp=2,
    ):
        super(ChiPKLin, self).__init__()
        self.num_comp = num_comp
        myokit.Unit.register('hour', myokit.Unit([0, 0, 1, 0, 0, 0, 0], np.log10(60**2)), quantifiable=True, output=True)

        # Create the model
        self._model = self.make_myokit_model()
        self.set_units()

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
        self.set_outputs(['central.drug_concentration'])
    
    def make_myokit_model(self):
        self.name = "Linear PK "+str(self.num_comp)+" compartment model"
        model = myokit.Model(name=self.name)
        
        cent_comp = model.add_component("central")
        cent_comp.add_variable("V_c").set_rhs(1)
        cent_comp.add_variable("K_cl").set_rhs(0)
        A_c = cent_comp.add_variable("drug_amount")
        C_c = cent_comp.add_variable("drug_concentration")
        
        A_c.promote()
        eq_A_c = "-K_cl*drug_concentration"
        C_c.set_rhs("drug_amount/V_c")

        for i in range(1, self.num_comp):
            comp_name = "periferal_"+str(i)
            perif_comp = model.add_component(comp_name)
            perif_comp.add_variable("K_p").set_rhs(0)
            perif_comp.add_variable("V_p").set_rhs(1)
            A_p = perif_comp.add_variable("drug_amount")
            C_p = perif_comp.add_variable("drug_concentration")

            A_p.promote()
            A_p.set_rhs("K_p*(central.drug_concentration - drug_concentration)")
            eq_A_c = eq_A_c+"-"+comp_name+".K_p*(drug_concentration - "+comp_name+".drug_concentration)"
            C_p.set_rhs("drug_amount/V_p")

        A_c.set_rhs(eq_A_c)
        time = model.add_component("time").add_variable("t")
        time.set_rhs(0)
        time.set_binding("time")

        model.validate(remove_unused_variables=False)
        return model

    def set_units(self, units={'time': 'hour', 'volume': 'L', 'amount': 'mg'}):
        time_unit = myokit.Unit.parse_simple(units['time'])
        volume_unit = myokit.Unit.parse_simple(units['volume'])
        amount_unit = myokit.Unit.parse_simple(units['amount'])

        self._model.time().set_unit(unit=time_unit)
        for comp in self._model.components():
            if not comp.name() == 'time':
                comp['drug_amount'].set_unit(unit=amount_unit)
                comp['drug_concentration'].set_unit(unit=amount_unit/volume_unit)
                if comp.name()=='central':
                    comp['V_c'].set_unit(unit=volume_unit)
                    comp['K_cl'].set_unit(unit=volume_unit/time_unit)
                else:
                    comp['V_p'].set_unit(unit=volume_unit)
                    comp['K_p'].set_unit(unit=volume_unit/time_unit)

        self._model.validate(remove_unused_variables=False)
        self._model.check_units()

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
        self._set_state([0]*self._n_states)

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
