import pints
import numpy as np
import pandas
from Code.Likelihoods import ConstantAndMultiplicativeGaussianLogLikelihood
from Code.PD_model import PintsPDFribergLinE as Pints_PD_model


def maximum_likelihood(
    problem, noise_model, start_point, boundaries=None, transform=False
):
    prev = [True]*7
    fix_noise = {
        'Combined': [None]*3,
        'Combined-fix eta': [None, 1, None],
        'Multiplicative': [0, None, None],
        'Multiplicative-fix eta': [0, 1, None],
        'Constant': [None, 1, 0]
    }
    unchanged_threshold = 1e-4

    if noise_model == "All":
        noise_model = fix_noise.keys()
    elif isinstance(noise_model, str):
        if noise_model in fix_noise.keys():
            noise_model = [noise_model]
        else:
            raise TypeError(
                'noise_model must be one of '+str(fix_noise.keys()+['All'])+
                ' or a list of these'
            )
    elif hasattr(noise_model, '__iter__'):
        if "All" in noise_model:
            noise_model = fix_noise.keys()
    else:
        raise TypeError(
            'noise_model must be one of '+str(fix_noise.keys()+['All'])+
            ' or a list of these'
        )

    results = {}
    for noise in noise_model:
        print("Optimising for "+noise+" noise model")
        log_likelihood = ConstantAndMultiplicativeGaussianLogLikelihood(
            problem, fix_noise=fix_noise[noise]
        )
        error_measure = pints.ProbabilityBasedError(log_likelihood)
        parameter_inclusion = np.asarray([None]*4+fix_noise[noise])==None

        
        if boundaries is None:
            noise_bound = None
        else:
            noise_bound = pints.RectangularBoundaries(
                np.asarray(boundaries)[0, parameter_inclusion], np.asarray(boundaries)[1, parameter_inclusion]
            )
            if transform:
                transformation = pints.RectangularBoundariesTransformation(noise_bound)
                # noise_bound = None
            else:
                transformation = None
        optimisation = pints.OptimisationController(
            log_likelihood,
            np.asarray(start_point)[parameter_inclusion],
            method=pints.CMAES,
            transformation=transformation,
            # boundaries=noise_bound
        )
        
        def cb(i, opt):
            params = transformation.to_model(opt.xbest())
            if noise == 'Multiplicative' or noise == 'Multiplicative-fix eta':
                params = np.insert(params, [4], None)
            if noise in ['Combined-fix eta', 'Multiplicative-fix eta', 'Constant']:
                params = np.insert(params, [5], None)
            if noise == 'Constant':
                params = np.insert(params, [6], None)
            nonlocal prev
            prev = np.all(np.asarray([prev, boundaries[0]<=params, params<=boundaries[1]]), axis=0)
            print("\r" + str(i) + str(prev),
                sep=' ',
                end='        ',
                flush=True
            )

        optimisation.set_callback(cb=cb)
        optimisation.set_max_unchanged_iterations(threshold=unchanged_threshold)
        optimisation.set_log_to_screen(False)
        parameters, error = optimisation.run()
        time = optimisation.time()
        print('\tOptimisation complete')
        params = parameters
        if noise == 'Multiplicative' or noise == 'Multiplicative-fix eta':
            params = np.insert(params, [4], None, axis=None)
        if noise in ['Combined-fix eta', 'Multiplicative-fix eta', 'Constant']:
            params = np.insert(params, [5], None, axis=None)
        if noise == 'Constant':
            params = np.insert(params, [6], None, axis=None)
        print('\tLog-Likelihood: \t'+str(-error))
        print('\tTime Taken: \t'+str(int(time/60))+" minutes, "+str(int(time%60))+" seconds, ")

        results[noise] = (parameters, error)
    return results

def bayesian_inference(
    problem, noise_model, start_point, boundaries, transform=False
):
    fix_noise = {
        'Combined': [None]*3,
        'Combined-fix eta': [None, 1, None],
        'Multiplicative': [0, None, None],
        'Multiplicative-fix eta': [0, 1, None],
        'Constant': [None, 1, 0]
    }
    unchanged_threshold = 1e-4 
    num_iterations = len(start_point)*5000  # 5000 iterations per parameter
    num_samples = len(start_point)*2250  # 2250 final samples per parameter

    if noise_model == "All":
        noise_model = fix_noise.keys()
    elif isinstance(noise_model, Iterable):
        if "All" in noise_model:
            noise_model = fix_noise.keys()
    else:
        if noise_model in fix_noise.keys():
            noise_model = [noise_model]
        else:
            raise TypeError(
                'noise_model must be one of '+str(fix_noise.keys()+['All'])+
                ' or a list of these'
            )

    results = {}
    for noise in noise_model:
        print("Performing MCMC for "+noise+" noise model")
        parameter_inclusion = np.asarray([None]*4+fix_noise[noise])==None

        if boundaries is None:
            raise TypeError(
                'You must provide boundaries for the uniform log-prior'
            )
        noise_bound = pints.RectangularBoundaries(
            np.asarray(boundaries)[0, parameter_inclusion], np.asarray(boundaries)[1, parameter_inclusion]
        )
        log_prior = pints.UniformLogPrior(noise_bound)
        start_points = [
            np.exp(np.log(np.asarray(boundaries)[0, parameter_inclusion])*0.25+np.log(np.asarray(start_point)[parameter_inclusion])*0.75), 
            np.asarray(start_point)[parameter_inclusion], 
            np.exp(np.log(np.asarray(boundaries)[1, parameter_inclusion])*0.25+np.log(np.asarray(start_point)[parameter_inclusion])*0.75)
        ]
        if transform:
            transformation = pints.RectangularBoundariesTransformation(noise_bound)
        else:
            transformation = None
        log_likelihood = ConstantAndMultiplicativeGaussianLogLikelihood(
            problem, fix_noise=fix_noise[noise]
        )
        samples = MCMC_run(
            log_likelihood,
            log_prior,
            start_points,
            num_iterations,
            num_samples,
            transform = transformation
            # save_point_like='../Data_and_parameters/PD_sim/MCMC_'+noise+'_pointloglike_chain'
        )
        results[noise] = (samples)
    return results

def MCMC_run(log_likelihood, log_prior, startpoints, max_iterations, num_samples, transform, save_point_like=None):
    
    hb_acmc = []
    for point in startpoints:
        hb_acmc.append(pints.HaarioBardenetACMC(point))
        hb_acmc[-1].set_initial_phase(True)
    num_initial = 0.1*num_samples
    list_sample = [[]]*len(startpoints)
    if save_point_like is not None:
        list_pointwise = [[]]*len(startpoints)
    final_iteration = max_iterations
    timer = pints.Timer()
    
    print("iter", end=' ')
    for chain in range(len(hb_acmc)):
        print('\t', "chain "+str(chain), end=' ')
    print('\t', "R_hat", '\t\t', "Time")

    i = 0
    x=-9
    r_hat = np.NaN
    
    while i<final_iteration:
        # update each chain using ask/tell and determine pointwise log_likelihoods
        for chain, alg in enumerate(hb_acmc):
            if i == num_initial:
                alg.set_initial_phase(False)
            theta_hat = alg.ask()
            try:
                pointwise = log_likelihood.create_pointwise_loglikelihoods(theta_hat)
                lp = np.sum(pointwise) + log_prior(theta_hat)
            except:
                print("Error for parameters:", theta_hat)
                raise
            theta_g, _, accepted = alg.tell(lp)
            list_sample[chain] = list_sample[chain]+[theta_g]
            if save_point_like is not None:
                if accepted:
                    list_pointwise[chain]=list_pointwise[chain]+[pointwise]
                else:
                    list_pointwise[chain] = list_pointwise[chain] +[list_pointwise[chain][-1]]
        if i == num_initial:
            print("...........................")
            print("End of Initial Phase")
            print("...........................")
        if i == num_initial+num_samples:
            print("...........................")
            print("Sart of Convergence Testing")
            print("...........................")
        
        # Have the chains converged?
        if i>=num_initial+num_samples:
            r_hat = pints.rhat(np.asarray(list_sample)[:, -num_samples:, :])
            if np.all(r_hat<=1.01):
                final_iteration=i
                print("complete")
        i+=1
        # Print Output
        if (i<=200 and i%20==0) or (i >= x):
            x = i+int(num_samples/20)
            print(str(i), end=' ')
            for alg in hb_acmc:
                print('\t',round(float(alg.acceptance_rate()),4),end='     ')
            print('\t', round(np.average(r_hat),4),end='     ')
            print('\t', timer.format(timer.time()))
    
    if save_point_like is not None:
        np.savez_compressed(save_point_like, *pointwise)
    return list_sample


if __name__=="__main__":
    df = pandas.read_csv("Data_and_parameters/PD_sim/sythesised_data_real_timepoints.csv")
    df = df.sort_values(['ID', 'TIME'], ascending=True, ignore_index=True)
    PK_params = np.load("Data_and_parameters/PK_sim/actual_params.npy")[1, :].astype('float64')
    df_before_0 = df[df["TIME"] < 0]
    R_0_approx = np.mean(df_before_0["OBS"])
    PD_model = Pints_PD_model(PK_params[:-1], data=df, num_comp=2)
    pseudo_times = PD_model.pseudotime
    problem = pints.SingleOutputProblem(PD_model, pseudo_times, df['OBS'].to_numpy())

    lower_bound = [0.1*R_0_approx, df['TIME'].max()*0.01, 0.005, 0.01,     0.001,      0.001, 0.001]
    upper_bound = [10*R_0_approx,      df['TIME'].max(),     5,    100,   R_0_approx,   10,    1]
    point = list(np.exp((np.log(np.asarray(lower_bound)) + np.log(np.asarray(upper_bound)))/2))
    point[0] = R_0_approx
    x = maximum_likelihood(problem, "All", point, boundaries=[lower_bound, upper_bound], transform=True)
    print(x)
