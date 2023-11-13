print("Loading modules")

import numpy as np
import pandas
import os
from plotly import figure_factory as ff
from Code.PK_model import ChiPKLin
import chi
import logging
import pints
import plotly.express.colors as pxclrs
from Code.Plot import Plot_Models
import plotly.graph_objects as go
from scipy import interpolate
import pickle

# V_c partially done

data_mix_param_sets = [["V_c", ], ["V_c", "K_cl"]]
model_mixed_params = ["V_c"]  # , "K_cl"
timer = pints.Timer()

data_set_numbers = [4, 5, 7] # range(0, 8)
for data_set in data_set_numbers:
    print("Starting analysis for data-set " + str(data_set+1))

    timer.reset()
    # Adjustables
    is_sim = True
    drug = 'large_data_' + str(data_set%4+1)
    num_PK_comp = 1
    observation_name = 'Platelets '
    noise_mult = 1
    data_mixed_params = data_mix_param_sets[int(data_set/4)] # ["V_c", "K_cl"]
    model_no_pooled_params = []
    model_fixed_params = ["K_1", "V_1", "sigma_PK"]

    print("Setting folders to save to")

    if is_sim:
        PK_data_file = "../Data_and_parameters/PK_sim/PK_comp"+str(num_PK_comp)+"/"
        PD_data_file = "../Data_and_parameters/PD_sim/PK_comp"+str(num_PK_comp)+"/"
    else:
        PK_data_file = "../Data_and_parameters/PK_real/"
        PD_data_file = "../Data_and_parameters/PD_real/"


    # Collect fixed and mixed effects params
    PK_params = ["K_cl", "V_c", "K_1", "V_1", "K_2", "V_2",  "sigma_PK"]
    PD_params = ["S", "R_0", "Nonsense", "gamma", "MTT", "sigma_PD"]
    all_params = (
        [PD_params[0]] + PK_params[:2*num_PK_comp] + PD_params[1:-1] + PK_params[-1:]
        + PD_params[-1:]
    )
    PK_params = PK_params[:2*num_PK_comp] + PK_params[-1:]

    PKPD_param_numbers = dict(zip(all_params, range(11)))
    PK_param_numbers = dict(zip(PK_params, range(6)))
    PKPD_pop_param_numbers = {}
    pos = 0
    for param in [x for x in all_params if (x not in model_fixed_params)]:
        PKPD_pop_param_numbers[param] = pos
        pos += 1
        if param in model_mixed_params:
            PKPD_pop_param_numbers["omega_"+param] = pos
            pos += 1

    PK_pop_param_numbers = {}
    pos = 0
    for param in PK_params:
        PK_pop_param_numbers[param] = pos
        pos += 1
        if param in model_mixed_params:
            PK_pop_param_numbers["omega_"+param] = pos
            pos += 1

    # File Name additions
    fixed_file = ""
    model_fixed_params = sorted(model_fixed_params)
    if len(model_fixed_params) > 0:
        fixed_file += "_fixed"
        for param in model_fixed_params:
            fixed_file += "_"+param

    PK_pop_model_file = ""
    if len(model_mixed_params) == 0 and len(model_no_pooled_params) == 0:
        PK_pop_model_file += "fixed_effects"
    else:
        PK_pop_model_file += "pop"
        for param in model_mixed_params:
            if param in PK_params:
                PK_pop_model_file += "_"+param
        if len(model_no_pooled_params) > 0:
            PK_pop_model_file += "_independ"
            for param in model_no_pooled_params:
                if param in PK_params:
                    PK_pop_model_file += "_"+param

    PD_pop_model_file = ""
    if len(model_mixed_params) == 0 and len(model_no_pooled_params) == 0:
        PD_pop_model_file += "fixed_effects"
    else:
        PD_pop_model_file += "pop"
        for param in model_mixed_params:
            PD_pop_model_file += "_"+param
        if len(model_no_pooled_params) > 0:
            PD_pop_model_file += "_independ"
            for param in model_no_pooled_params:
                PD_pop_model_file += "_"+param

    if is_sim:
        if len(data_mixed_params)==0:
            drug += "_no_pop"
        else:
            drug += "_"+str(len(data_mixed_params))+"pop"
        if noise_mult != 1:
            drug += "_small_noise_"+str(noise_mult)


    PK_image_file = "../Images/PK_sim/PK_comp"+str(num_PK_comp)+"/"+drug
    PD_image_file = "../Images/PD_sim/PD_comp"+str(num_PK_comp)+"/"+drug
    mono_file = "../Monolix/PK_sim/PK_comp"+str(num_PK_comp)+"/"+drug

    # os.makedirs(PK_data_file+drug, exist_ok=True)
    # os.makedirs(PD_data_file+drug, exist_ok=True)
    # os.makedirs(PK_image_file, exist_ok=True)
    # os.makedirs(PD_image_file, exist_ok=True)


    def create_PK_true_param():
        PK_comp2_true_typs = np.load("/home/rumney/Documents/Myleotoxicity/Myleotoxicity-PKPD/Data_and_parameters/PK_sim/PK_comp2/actual_params.npy")[1]
        PK_comp2_true_typs = PK_comp2_true_typs.astype(float)

        if num_PK_comp == 1:
            PK_comp_vols = [['V_c, L'], [PK_comp2_true_typs[1]+PK_comp2_true_typs[3]]]
        else:
            PK_comp_vols = [
                ['V_c, L']+['V_'+str(x)+', L' for x in range(1, num_PK_comp)],
                [PK_comp2_true_typs[1]]+[PK_comp2_true_typs[3]/num_PK_comp]*num_PK_comp
            ]
        PK_rates = [
            ['K_{cl}, L/hr'] + ['K_'+str(x)+', L/hr' for x in range(1, num_PK_comp)],
            [PK_comp2_true_typs[0]] + [PK_comp2_true_typs[3]/x for x in range(1, num_PK_comp)]
        ]

        PK_true_typs = np.empty((2, 2*num_PK_comp+1), dtype='<U32')
        PK_true_typs[:, 0:-1:2] = PK_rates
        PK_true_typs[:, 1:-1:2] = PK_comp_vols
        PK_true_typs[:, -1] = ['sigma_{m, PK}', PK_comp2_true_typs[-1]]

        # PK_true_typs[1, 0] = 0.5*float(PK_true_typs[1, 0])
        np.save(PK_data_file+"actual_params.npy", PK_true_typs)

        PK_true_omegas = [0.3]*(num_PK_comp+1)
        np.save(PK_data_file+"omega.npy", PK_true_omegas)

        return PK_true_typs, PK_true_omegas

    print("Setting parameters")
    try:
        # Load the Data-generating typical parameters
        PK_true_typs = np.load(PK_data_file+"actual_params.npy")

        # Load the Data-generating inter-individual variability parameters
        PK_true_omegas = np.load(PK_data_file+"omega.npy")
    except FileNotFoundError:
        PK_true_typs, PK_true_omegas = create_PK_true_param()

    split_param_names = np.char.split(PK_true_typs[0, :], ", ")
    PK_true_pops = PK_true_typs
    for param in PK_params:
        pos_typ = PK_param_numbers[param]
        pos_pop = PK_pop_param_numbers[param]
        if param in data_mixed_params:
            omega = PK_true_omegas[pos_typ]
        else:
            omega = 1e-5

        if param in model_mixed_params:
            PK_true_pops = np.insert(
                PK_true_pops,
                pos_pop+1,
                ["omega_{"+split_param_names[pos_typ][0]+"}", omega],
                axis=1
            )
            PK_true_pops[0, pos_pop] = str().join(
                [split_param_names[pos_typ][0], "_typ", ", "]
                + list(split_param_names[pos_typ][1:])
            )

    PK_true_pops[1, -1] = noise_mult*float(PK_true_pops[1, -1])
    # PK_bounds = np.load(PK_data_file+"bounds.npy")

    table_df = pandas.DataFrame(
        PK_true_pops.transpose(),
        columns=['Parameter name', 'Data generating value']
    )
    table_df = table_df.astype({'Data generating value': float})

    table_df = table_df.round({'Data generating value': 4})
    table_df = table_df.set_index('Parameter name').transpose()
    fig = ff.create_table(table_df)
    fig.update_layout(
        width=500,
        height=45,
    )

    pop_param_names = PK_true_pops[0, :]
    PK_true_pops = PK_true_pops[1, :].astype('float64')

    PK_param_names = PK_true_typs[0, :]
    PK_true_typs = PK_true_typs[1, :].astype('float64')

    # fig.write_image(PK_image_file+"/true_values.svg")
    # fig.show()


    # Remove Annoying logging
    logger = logging.getLogger()
    logger.handlers = []

    print("Setting the model")
    # Set up the model
    PK_model = ChiPKLin(num_comp=num_PK_comp)
    PK_model.set_administration(
        compartment='central', amount_var='drug_amount')
    PK_model.set_outputs(['central.drug_concentration'])
    noise_model = chi.MultiplicativeGaussianErrorModel()
    n_noise = noise_model.n_parameters()
    chi_param_names = np.concatenate((PK_model.parameters(), noise_model.get_parameter_names()))

    pop_models = []
    for i_param, param in enumerate(PK_params):
        chi_param = chi_param_names[i_param]
        if param in model_mixed_params:
            print(chi_param, "mixed")
            pop_models.append(chi.LogNormalModel(n_dim=1))
        elif param in model_no_pooled_params:
            print(chi_param, "no pooled")
            pop_models.append(chi.HeterogeneousModel(n_dim=1))
        else:
            print(chi_param, "pooled")
            pop_models.append(chi.PooledModel(n_dim=1))

    population_model = chi.ComposedPopulationModel(pop_models)

    # Set up the inference problem
    problem = chi.ProblemModellingController(PK_model, noise_model)
    problem.set_population_model(population_model)


    def create_PK_dataset(dose_amts, n_ids_per_dose, times, dose_time=0):
        n_ids_data = n_ids_per_dose*len(dose_amts)

        # Get log of mixed effect typical parameters
        pop_params = PK_true_pops.copy()
        for param in PK_params:
            if param in model_mixed_params:
                pos = PK_pop_param_numbers[param]
                pop_params[pos] = np.log(pop_params[pos])
        
        # Acquire patient parameters
        individual_parameters = population_model.sample(
            parameters=pop_params,
            n_samples=n_ids_data
        )

        # Set up Dataframe
        df = pandas.DataFrame(columns=[
            "ID", "Time", "Observable", "Value", "Dose group", "Duration", "Dose"
        ])
        # Generate data
        for i_dose, dose in enumerate(dose_amts):
            PK_model.set_dosing_regimen(dose=dose, start=dose_time, period=0)
            for i_ind in range(0, n_ids_per_dose):
                # Simulate model
                pat_param = individual_parameters[i_ind+i_dose, :]
                patient_result = PK_model.simulate(pat_param[:-n_noise], times)
                patient_result = noise_model.sample(
                    pat_param[-n_noise:], patient_result[0, :]
                )[:, 0]

                # Format patient data
                patient_data= pandas.DataFrame(columns=[
                    "ID", "Time", "Observable", "Value", "Dose group", "Duration", "Dose"
                ])
                patient_id = i_ind+(i_dose*n_ids_per_dose)+1
                patient_data["ID"] = [patient_id]*(len(times)+1)
                patient_data["Time"] = np.concatenate((times, [dose_time]))
                patient_data["Observable"] = (
                    ['central.drug_concentration']*len(times)+[None]
                )
                patient_data["Value"] = np.concatenate((patient_result, [None]))
                patient_data["Dose group"] = [dose]*(len(times)+1)
                patient_data["Duration"] = [None]*len(times)+[0.01]
                patient_data["Dose"] = [None]*len(times)+[dose]

                # Join to main dataframe
                df = pandas.concat([df, patient_data])
            df = df.reset_index(drop=True)

        return df, individual_parameters


    if is_sim:
        print("Loading Data")
        PK_true_ind = {}
        # Check whether Dataset already exists
        try:
            df = pandas.read_csv(PD_data_file + drug+"/data.csv")
            df.drop(
                df.loc[df['Observable'] == 'circulating.R'].index, inplace=True
            )
            dose_time = (df.loc[df['Dose'].notna()])['Time'].min()
            df['Time'] = df['Time']-dose_time
            df.replace(
                {'PK_central.drug_concentration': 'central.drug_concentration'},
                inplace=True
            )
            for param in data_mixed_params:
                pos = PK_param_numbers[param]
                PK_true_ind[param] = np.load(PD_data_file + drug + "/ind_" + param + ".npy")
        except FileNotFoundError:
            try:
                df = pandas.read_csv(PK_data_file + drug+"/data.csv")
                os.makedirs(mono_file, exist_ok=True)
                df_mono = df.copy()
                df_mono.replace(
                    {'central.drug_concentration': 'central_drug_conc'},
                    inplace=True
                )
                df_mono.to_csv(mono_file + "/data.csv", index=False)
                for param in data_mixed_params:
                    pos = PK_param_numbers[param]
                    PK_true_ind[param] = np.load(PK_data_file + drug + "/ind_" + param + ".npy")

            except FileNotFoundError:
                print("No datafile found")
                print("Creating new dataset")
                # Select options for data
                dose_amts = [1.0, 2.0, 3.0]
                n_ids_per_dose = 15
                n_times_data = 50

                # Create and save simulated data
                times = np.linspace(0.05, 5, n_times_data)
                df, ind_params = create_PK_dataset(
                    dose_amts, n_ids_per_dose, times
                )
                df.to_csv(PK_data_file + drug+"/data.csv", index=False)

                os.makedirs(mono_file, exist_ok=True)
                df_mono = df.copy()
                df_mono.replace(
                    {'central.drug_concentration': 'central_drug_conc'},
                    inplace=True
                )
                df_mono.to_csv(mono_file + "/data.csv", index=False)
                for param in data_mixed_params:
                    pos = PK_param_numbers[param]
                    np.save(
                        PK_data_file + drug + "/ind_" + param + ".npy",
                        ind_params[:, pos]
                    )
                    PK_true_ind[param] = ind_params[:, pos]

    else:
        # Load in the data
        df = pandas.read_csv(PK_data_file + drug+"/data.csv")
        dose_unit = "mg"

        # Reformat this into Something Chi understands
        df = df.rename(columns={
            'TIME': 'Time', 'OBS': 'Value', 'DOSE': 'Dose group'
        })
        df.replace(
            {drug: 'central.drug_concentration'},
            inplace=True
        )
        dosing_df = df.groupby('ID', as_index=False)['Dose group'].mean()
        dosing_df.columns = ['ID', 'Dose']
        dosing_df['Time'] = [0.0]*len(dosing_df)
        dosing_df['Duration'] = [0.001]*len(dosing_df)

        df = pandas.concat([df, dosing_df], join='outer', ignore_index=True)

    df_obs = df.loc[df['Observable'] == 'central.drug_concentration']
    df = df.loc[df['ID'].isin(df_obs['ID'].unique())]
    n_ids_data = len(df["ID"].unique())

    dose_unit = "mg"
    problem.set_data(df)

    df


    from scipy.integrate import simpson
    from scipy.stats import linregress

    print("Finding initial approximations of parameters")
    # Find the first and second points for each individual
    df_dose = np.asarray(df.groupby('ID')['Dose'].first())
    df_obs['rank'] = df_obs.sort_values('Time').groupby('ID').cumcount()+1
    points_1st = df_obs[df_obs['rank'] == 1].sort_values('ID', ignore_index=True)
    points_2nd = df_obs[df_obs['rank'] == 2].sort_values('ID', ignore_index=True)

    # Determine y-intercept
    y_0 = points_1st['Value'] - points_1st['Time'] * (
        (points_1st['Value'] - points_2nd['Value'])
        / (points_1st['Time'] - points_2nd['Time'])
    )
    # Estimate V_c
    V_c_approx = (df_dose/y_0).mean()

    # Determine the AUC from the first to last point,
    AUC_0_last = np.empty(len(df["ID"].unique()))
    # and the last drug concentration value,
    C_last = np.empty(len(df["ID"].unique()))
    # and the rate of decay of the drug,
    lambda_z = np.empty(len(df["ID"].unique()))

    for i, patient in enumerate(df["ID"].unique()):
        y_ind = np.asarray(df_obs.loc[df_obs['ID'] == patient]['Value'])
        x_ind = np.asarray(df_obs.loc[df_obs['ID'] == patient]['Time'])
        AUC_0_last[i] = simpson(y=y_ind, x=x_ind)
        C_last[i] = y_ind[-1]
        lambda_z[i] = linregress(
            x=x_ind[int(0.5*len(x_ind)):],
            y=x_ind[int(0.5*len(x_ind)):]
        ).slope

    # Extrapolate AUC_0_last to infinity
    AUC_inf = AUC_0_last+C_last/lambda_z
    # Approximate the clearance
    cl_approx = (df_dose/AUC_inf).mean()

    Approximations = [
            cl_approx,          # Clearance
            V_c_approx          # Central volume
        ]+[
            cl_approx,          # Periferal compartment transfer
            V_c_approx          # Periferal compartment volume
        ]*(num_PK_comp-1)+[
            0.1,                # PK Noise parameter
    ]

    table_df.loc["Approximate"] = [np.nan]*len(table_df.columns)
    for i, param in enumerate(PK_params):
        pos = PK_pop_param_numbers[param]
        table_df.iat[1, pos] = Approximations[i]
    table_df = table_df.round(4)

    fig = ff.create_table(table_df, index=True)
    fig.update_layout(
        width=500,
        height=65
    )
    # fig.write_image(PK_image_file+"/data_table.svg")
    # fig.show()

    # Determine the shape parameters for the priors
    shape_parameters = [
            0.3,                       # Clearance
            0.3,                        # Central volume
        ]+[
            3,                          # Periferal compartment transfer
            0.6,                        # Periferal compartment volume
        ]*(num_PK_comp-1)+[
            0.4,                        # PK Noise parameter
    ]


    print("Building priors")
    # Build the priors and transformation of the parameter space
    log_priors = []
    transformations = []
    for param in PK_params:
        if param in model_mixed_params:
            # Prior for the typical value
            approx = np.log(Approximations[PK_param_numbers[param]])
            shape = shape_parameters[PK_param_numbers[param]]
            log_priors.append(pints.GaussianLogPrior(approx, shape))
            # Prior for omega
            approx = np.log(shape_parameters[PK_param_numbers[param]])
            log_priors.append(pints.LogNormalLogPrior(approx, 0.2))
            # Transformation for the individual parameters
            ind_trans = pints.LogTransformation(n_ids_data)
            transformations = [ind_trans] + transformations
            # Transformation for the population parameters
            transformations.append(pints.IdentityTransformation(2))
        else:
            # Prior and transformation for the pooled parameter
            approx = np.log(Approximations[PK_param_numbers[param]])
            shape = shape_parameters[PK_param_numbers[param]]
            log_priors.append(pints.LogNormalLogPrior(approx, shape))
            transformations.append(pints.LogTransformation(1))

    # Compose the priors together
    log_prior = pints.ComposedLogPrior(*log_priors)
    problem.set_log_prior(log_prior)

    # Compose the transformations together
    transformation = pints.ComposedTransformation(*transformations)


    #### Likelihood over the Parameter Space
    print("Building likelihood")
    log_posterior = problem.get_log_posterior()
    log_likelihood = log_posterior.get_log_likelihood()

    print("Loading PINTS MPE")
    opt_ind_params = np.load(PK_data_file+drug+'/'+PK_pop_model_file+"_opt_ind.npy")
    summary_df = pandas.read_csv(
        PK_data_file+drug+'/'+PK_pop_model_file+"_opt_pop.csv"
    )
    summary_df = summary_df.set_index("Parameter")

    log_posterior_values = summary_df.loc["Log-posterior"].drop(["Mean", "True"])
    log_posterior_values = np.asarray(log_posterior_values)

    best = log_posterior_values==(np.max(log_posterior_values))
    pop_best_of_converged = np.asarray(summary_df.iloc[:-1, :-2].loc[:, best]).flatten()
    ind_best_of_converged = opt_ind_params[best].flatten()

    ref_param = np.concatenate((ind_best_of_converged, pop_best_of_converged))

    print("Loading Monolix MLE")
    mono_runs = np.loadtxt(mono_file+'/MLE_'+PK_pop_model_file+'.csv')

    colour_arg = int(drug.split("_")[2]) + 4*(len(data_mixed_params)-1) - 1
    model_colour = pxclrs.qualitative.Safe[colour_arg]

    print("Initialising plots")
    plot = Plot_Models(
        pop_model=population_model,
        error_models=noise_model,
        mech_model=PK_model,
        data=df
    )
    plot.set_colour({"base":model_colour})
    lower_bounds = []
    upper_bounds = []
    n_mix = len(model_mixed_params)*n_ids_data

    for param in PK_params:
        pos = PK_pop_param_numbers[param]
        shape_pos = PK_param_numbers[param]
        param_value = ref_param[pos+n_mix]
        if param in model_mixed_params:
            omega_value = np.log(ref_param[pos+n_mix+1])
            lower_bounds.append(param_value-shape_parameters[shape_pos])
            lower_bounds.append(np.exp(omega_value-0.4))
            upper_bounds.append(param_value+shape_parameters[shape_pos])
            upper_bounds.append(np.exp(omega_value+0.4))
        else:
            param_value = np.log(param_value)
            lower_bounds.append(np.exp(param_value-shape_parameters[shape_pos]))
            upper_bounds.append(np.exp(param_value+shape_parameters[shape_pos]))

    names = pop_param_names

    # Mixed V_c, K_cl:
    if model_mixed_params == ["V_c", "K_cl"]:
        force_bound_values = {
            0:[(0, 1, 1e-4)], 1:[(0, 1, 1e-4)], 2:[(0, 1, 1e-4)], 3:[(0, 1, 1e-4)],
            4:[], 5:[], 6:[], 7:[]
        }
    # Mixed V_c: # 5, 7
    elif model_mixed_params == ["V_c"]:
        force_bound_values = {
            0:[(0, 0, 2.775), (1, 0, 2.815)], 1:[(0, 0, 2.79), (1, 0, 2.825)],
            2:[(0, 0, 2.78 ), (1, 0, 2.82), (1, 2, 0.475)], 3:[(0, 0, 2.782), (1, 0, 2.817)],
            4:[(0, 0, 2.66), (1, 0, 2.76)],
            5:[(0, 0, 2.43), (1, 0, 2.56), (0, 1, 1.2), (1, 2, 0.85)],
            6:[(0, 0, 2.34), (1, 0, 2.45), (1, 2, 0.75)], 7:[(0, 0, 2.51), (1, 0, 2.585), (1, 2, 0.62)]
        }
    force_bounds = [
        [False, False, False, False],
        [False, False, False, False]
    ]
    for fbv in force_bound_values[data_set]:
        if fbv[0] == 0:
            lower_bounds[fbv[1]] = fbv[2]
        else:
            upper_bounds[fbv[1]] = fbv[2]
        force_bounds[fbv[0]][fbv[1]] = True

    # Mixed K_cl:
    # lower_bounds[2] =   # 1: 5.725  # 2: 5.9  # 3: 5.4  # 4: 5.6  # 5: 5.25  # 6: 5  # 7: 4.9  # 8: 5.1
    # lower_bounds[1] =   # 1: 0.06  # 2: 0.06  # 3: 0.08  # 4: 0.075
    # upper_bounds[2] =   # 1: 5.92  # 2: 6.1  # 3: 5.7  # 4: 5.9  # 5: 5.425  # 6: 5.275  # 7: 5.125  # 8: 5.4
    # force_bounds=(
    #     [False, False, True, False],
    #     [False, False, True, False]
    # )

    # Fixed:
    # lower_bounds[0] =   # 1: 2.82  # 2: 2.83  # 3: 2.85  # 4: 2.82  # 7: 2.775
    # upper_bounds[0] =   # 1: 2.91  # 2: 2.905  # 3: 2.945  # 4: 2.925  # 5: 3.1
    # upper_bounds[1] =   # 5: 6.65
    # force_bounds=(
    #     [False, False, False, False],
    #     [True, True, False, False]
    # )


    print("Finding and plotting profile likelihoods")

    fig = plot.plot_param_function(
        log_likelihood, ref_param, profile="maximum", pairwise=False,
        individual_parameters=False, param_names=names, bounds=(lower_bounds, upper_bounds),
        force_bounds=force_bounds, n_evals=100
    )

    print("Adding MLEs to plot")
    opt_score = log_likelihood(ref_param)

    min_x = [np.inf]*(len(ref_param)-n_mix)
    max_x = [-np.inf]*(len(ref_param)-n_mix)
    min_y = [np.inf]*(len(ref_param)-n_mix)
    max_y = [-np.inf]*(len(ref_param)-n_mix)
    interpolator = [None]*(len(ref_param)-n_mix)
    for trace_data in fig.data:
        param_arg = trace_data.xaxis.split("x")[-1]
        if param_arg == "":
            param_arg = 0
        else:
            param_arg = int(param_arg)-1
        if len(trace_data.x)>2:
            min_x[param_arg] = min(trace_data.x)
            max_x[param_arg] = max(trace_data.x)
            
            min_y[param_arg] = min(trace_data.y)
            max_y[param_arg] = max(trace_data.y)

            interpolator[param_arg] = interpolate.interp1d(trace_data.x, trace_data.y)

    cmaes_runs = np.transpose(np.asarray(summary_df.iloc[:-1, :-2]))

    for cmaes_run, cmaes_inf_param in enumerate(cmaes_runs):
        # if best[cmaes_run]:
        #     continue

        for param_arg, x in enumerate(cmaes_inf_param):
            row = int(param_arg/3)+1
            col = param_arg%3+1

            if min_x[param_arg]<x<max_x[param_arg]:
                cmaes_param_y = interpolator[param_arg](x)
                if cmaes_param_y<-3.5:
                    cmaes_param_y = max_y[param_arg]
            else:
                cmaes_param_y = max_y[param_arg]

            fig.add_trace(
                go.Scatter(
                    name="Data-Set "+str(colour_arg+1),
                    y=[min_y[param_arg], cmaes_param_y],
                    x=[x]*2,
                    mode='lines',
                    line=dict(color=model_colour, width=2),
                    showlegend=(param_arg == 0)&(cmaes_run == 0),
                    legendgroup="cmaes",
                    legendgrouptitle_text="CMA-ES run result"
                ),
                row=row,
                col=col
            )
        fig.update_yaxes(
            range=[min_y, max_y],
            row=row,
            col=col
        )

    for mono_run, mono_inf_param in enumerate(mono_runs):
        for param_arg, x in enumerate(mono_inf_param):
            if x is None:
                continue
            row = int(param_arg/3)+1
            col = param_arg%3+1

            param = list(PK_pop_param_numbers.keys())[param_arg]
            if param in model_mixed_params:
                x = np.log(x)
            
            if min_x[param_arg]<x<max_x[param_arg]:
                mono_param_y = interpolator[param_arg](x)
                if mono_param_y<-3.5:
                    mono_param_y = max_y[param_arg]
            else:
                mono_param_y = max_y[param_arg]

            fig.add_trace(
                go.Scatter(
                    name="Data-Set "+str(colour_arg+1),
                    y=[min_y[param_arg], mono_param_y],
                    x=[x]*2,
                    mode='lines',
                    line=dict(color=model_colour, width=2, dash='dot'),
                    showlegend=(param_arg == 0)&(mono_run == 0),
                    legendgroup="mononlix",
                    legendgrouptitle_text="Mononlix run result"
                ),
                row=row,
                col=col
            )

    print("Saving Image")
    with open(PK_image_file+'/'+PK_pop_model_file+"_ll_profiles_data.pkl", "wb") as fp:
        pickle.dump(fig.data, fp)  # encode dict into pickle
    fig.write_image(PK_image_file+'/'+PK_pop_model_file+"_ll_profiles_compare.svg")
    fig.show()

    # fig = plot.plot_param_function(
    #     log_likelihood, ref_param, profile="maximum", pairwise=True,
    #     individual_parameters=False, param_names=names, bounds=(lower_bounds, upper_bounds),
    #     force_bounds=force_bounds, n_evals=50
    # )
    # fig.write_image(PK_image_file+"/"+PD_pop_model_file+fixed_file+"_quick_ll_profiles_pair_from_True.svg")
    # fig.show()

    print("Done, time taken: " + timer.format(timer.time()))
