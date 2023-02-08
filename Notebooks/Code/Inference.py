import pints
import numpy as np
import pandas
import chi
from tqdm.notebook import tqdm
import xarray as xr
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
                'noise_model must be one of ' +
                str(fix_noise.keys() + ['All']) +
                ' or a list of these'
            )
    elif hasattr(noise_model, '__iter__'):
        if "All" in noise_model:
            noise_model = fix_noise.keys()
    else:
        raise TypeError(
            'noise_model must be one of ' + str(fix_noise.keys() + ['All']) +
            ' or a list of these'
        )

    results = {}
    for noise in noise_model:
        print("Optimising for "+noise+" noise model")
        log_likelihood = ConstantAndMultiplicativeGaussianLogLikelihood(
            problem, fix_noise=fix_noise[noise]
        )
        # error_measure = pints.ProbabilityBasedError(log_likelihood)
        parameter_inclusion = np.asarray([None]*4+fix_noise[noise]) == None

        if boundaries is None:
            noise_bound = None
        else:
            noise_bound = pints.RectangularBoundaries(
                np.asarray(boundaries)[0, parameter_inclusion],
                np.asarray(boundaries)[1, parameter_inclusion]
            )
            if transform:
                transformation = pints.RectangularBoundariesTransformation(
                    noise_bound)
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
            if noise in [
                'Combined-fix eta', 'Multiplicative-fix eta', 'Constant'
            ]:
                params = np.insert(params, [5], None)
            if noise == 'Constant':
                params = np.insert(params, [6], None)
            nonlocal prev
            prev = np.all(np.asarray(
                [prev, boundaries[0] <= params, params <= boundaries[1]]
            ), axis=0)
            print(
                "\r" + str(i) + str(prev), sep=' ', end='        ', flush=True
            )

        optimisation.set_callback(cb=cb)
        optimisation.set_max_unchanged_iterations(
            threshold=unchanged_threshold)
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
        print(
            '\tTime Taken: \t' + str(int(time/60)) + " minutes, " +
            str(int(time % 60)) + " seconds, "
        )

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
    # unchanged_threshold = 1e-4 
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
                'noise_model must be one of ' +
                str(fix_noise.keys() + ['All']) +
                ' or a list of these'
            )

    results = {}
    for noise in noise_model:
        print("Performing MCMC for "+noise+" noise model")
        parameter_inclusion = np.asarray([None]*4+fix_noise[noise]) == None

        if boundaries is None:
            raise TypeError(
                'You must provide boundaries for the uniform log-prior'
            )
        noise_bound = pints.RectangularBoundaries(
            np.asarray(boundaries)[0, parameter_inclusion],
            np.asarray(boundaries)[1, parameter_inclusion]
        )
        log_prior = pints.UniformLogPrior(noise_bound)
        start_points = [
            np.exp(
                np.log(np.asarray(boundaries)[0, parameter_inclusion])*0.25 +
                np.log(np.asarray(start_point)[parameter_inclusion])*0.75
            ),
            np.asarray(start_point)[parameter_inclusion],
            np.exp(
                np.log(np.asarray(boundaries)[1, parameter_inclusion])*0.25 +
                np.log(np.asarray(start_point)[parameter_inclusion])*0.75
            )
        ]
        if transform:
            transformation = pints.RectangularBoundariesTransformation(
                noise_bound)
        else:
            transformation = None
        log_likelihood = ConstantAndMultiplicativeGaussianLogLikelihood(
            problem, fix_noise=fix_noise[noise]
        )
        log_post = pints.LogPosterior(log_likelihood, log_prior)
        samples = SamplingController(log_post).MCMC_run(
            start_points,
            num_iterations,
            num_samples,
            transform=transformation
            # save_point_like='../Data_and_parameters/PD_sim/MCMC_'+noise+'_pointloglike_chain'
        )
        results[noise] = (samples)
    return results


class SamplingController(chi.InferenceController):

    """
    Sets up a sampling routine that attempts to find the posterior
    distribution of parameters defined by a :class:`pints.LogPosterior`. If
    multiple log-posteriors are provided, the posteriors are assumed to be
    structurally identical and only differ due to different data sources.

    By default the sampling is run 5 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.

    :param log_posterior: The log-posterior from which is sampled.
    :type log_posterior: chi.LogPosterior, chi.HierarchicalLogPosterior
    :param seed: Seed for the random initialisation. Initial points are sampled
        from the log-prior.
    :type seed: int
    """

    def __init__(self, log_posterior, seed=None):
        super(SamplingController, self).__init__(log_posterior, seed)

        # Set default sampler
        self._sampler = pints.HaarioBardenetACMC

    def set_sampler(self, sampler):
        """
        Sets method that is used to sample from the log-posterior.
        """
        if not issubclass(sampler, pints.MCMCSampler):
            raise ValueError(
                'Sampler has to be a `pints.MCMCSampler`.'
            )
        self._sampler = sampler

    def make_initial_point(
        self, n_runs, population_parameters, individual_parameters=None
    ):
        log_post = self._log_posterior
        n_top = log_post.n_parameters(exclude_bottom_level=True)
        n_bottom = log_post.n_parameters(exclude_bottom_level=False) - n_top
        log_likelihood = log_post.get_log_likelihood()

        population_parameters = np.atleast_2d(population_parameters)
        if population_parameters.shape[1] != n_top:
            raise ValueError(
                'population parameters must be same length as the ' +
                'log-posterior population parameters')

        population_model = log_likelihood.get_population_model()
        dims = []
        current_dim = 0
        
        if individual_parameters is not None:
            individual_parameters = np.atleast_2d(individual_parameters)
            if individual_parameters.shape[1] != n_bottom:
                raise ValueError(
                    'individual parameters must match the ' +
                    'number of individuals')

        if isinstance(population_model, chi.ReducedPopulationModel):
            population_model = population_model.get_population_model()
        try:
            pop_models = population_model.get_population_models()
        except AttributeError:
            pop_models = [population_model]
        for pop_model in pop_models:
            n_dim = pop_model.n_dim()
            if isinstance(
                    pop_model, (chi.PooledModel, chi.HeterogeneousModel)):
                current_dim += n_dim
                continue
            end_dim = current_dim + n_dim
            dims += list(range(current_dim, end_dim))
            current_dim = end_dim

        initial_points = np.empty((n_runs, n_top+n_bottom))

        for i in range(0, n_runs):
            if individual_parameters is None:
                n_ids = log_likelihood.n_log_likelihoods()
                covariates = log_likelihood._covariates
                bottom_parameters = []
                population_model = log_likelihood.get_population_model()
                bottom_parameters.append(population_model.sample(
                    parameters=population_parameters[i, :],
                    n_samples=n_ids, covariates=covariates))
                # Remove pooled dimensions
                # (Pooled and heterogen. dimensions do not count as bottom 
                # parameters)
                for idx, bottom_params in enumerate(bottom_parameters):
                    bottom_parameters[idx] = bottom_params[:, dims].flatten()
                bottom_parameters = np.vstack(bottom_parameters)
            else:
                bottom_parameters = individual_parameters[i, :]

            # print(self._initial_params[run-1, n_bottom:])
            initial_points[i, n_bottom:] = population_parameters[i, :]
            initial_points[i, :n_bottom] = bottom_parameters
        return initial_points

    def set_initial_point(
        self, run_ids, population_parameters, individual_parameters=None
    ):
        """
        Sets the initial starting points for a selection of the runs.
        """
        if isinstance(run_ids, int):
            run_ids = [run_ids]
        run_ids = np.asarray(run_ids)
        if any(run_ids > self._n_runs) or any(run_ids <= 0):
            raise ValueError(
                'run_ids must be an int or list of int between 1 and the ' +
                'currently set number of runs. Use set_n_runs to increase ' +
                'number of runs set')

        initial_points = self.make_initial_point(
            len(run_ids), population_parameters, individual_parameters
        )
        for i, run in enumerate(run_ids):
            self._initial_params[run-1, :] = initial_points[i, :]

    def _format_chains(self, chains, divergent_iters):
        """
        Formats the chains generated by pints in shape of
        (n_chains, n_iterations, n_parameters) to a xarray.Dataset, where
        each parameter name gets an entry yielding a xarray.DataArray of
        shape (n_chains, n_iterations, n_ids) or if ID is None
        (n_chains, n_iterations).

        Note that the naming of the dimensions matter for ArviZ so we call
        the dimensions (chain, draw, individual) (the last dimension
        is not set by ArviZ).
        """

        # Get all parameter names
        names = self._log_posterior.get_parameter_names()

        # Get top-level parameter names
        top_parameters = self._log_posterior.get_parameter_names(
            exclude_bottom_level=True)

        # Get IDs of bottom-level parameters
        ids = self._log_posterior.get_id(unique=True)

        # Convert names and ids to numpy arrays
        ids = np.array(ids) if isinstance(ids, list) else ids
        names = np.array(names)

        # Get the coordinates for the chains and draws
        n_chains, n_draws, _ = chains.shape
        chain_coords = list(range(n_chains))
        draw_coords = list(range(n_draws))

        # Sort samples of parameters into xarrays
        # Start with top-level parameters (have no ID)
        container = {}
        bottom_parameters = []
        for idp, parameter in enumerate(names):
            # Check if parameter is a bottom-level parameter
            if parameter not in top_parameters:
                # Add to bottom parameter list if parameter is not already in
                # it
                if parameter not in bottom_parameters:
                    bottom_parameters.append(parameter)
                continue

            # Append top parameter samples to container
            container[parameter] = xr.DataArray(
                data=chains[:, :, idp],
                dims=['chain', 'draw'],
                coords={'chain': chain_coords, 'draw': draw_coords})

        # Add samples of bottom parameters (include IDs now)
        for idp, parameter in enumerate(bottom_parameters):
            mask = names == parameter
            container[parameter] = xr.DataArray(
                data=chains[:, :, mask],
                dims=['chain', 'draw', 'individual'],
                coords={
                    'chain': chain_coords,
                    'draw': draw_coords,
                    'individual': ids})

        # Add information about divergent iterations
        attrs = {'divergent iterations': 'false'}
        if divergent_iters:
            attrs['divergent iterations'] = 'true'
            for idx, iters in enumerate(divergent_iters):
                attrs['divergent iterations chain %d' % idx] = iters

        return xr.Dataset(container, attrs=attrs)

    def run(
        self, n_iterations, sigma0=None, hyperparameters=None
    ):
        """
        Runs the sampling routine and returns the sampled parameter values in
        form of a :class:`xarray.Dataset` with :class:`xarray.DataArray`
        instances for each parameter.

        If multiple posteriors are inferred a list of :class:`xarray.Dataset`
        instances is returned.

        The number of iterations of the sampling routine can be set by setting
        ``n_iterations`` to a finite, non-negative integer value. By default
        the routines run for 10000 iterations.

        :param n_iterations: A non-negative integer number which sets the
            number of iterations of the MCMC runs.
        :type n_iterations: int, optional
        :param hyperparameters: A list of hyperparameters for the sampling
            method. If ``None`` the default hyperparameters are set.
        :type hyperparameters: list[float], optional
        :param log_to_screen: A boolean flag which can be used to print
            the progress of the runs to the screen. The progress is printed
            every 500 iterations.
        :type log_to_screen: bool, optional
        """

        # Configure sampling routine
        # self._log_to_screen(log_to_screen)
        # self._log_interval(iters=20, warm_up=3)
        self._required_iters = n_iterations

        # Run sampling routine
        chains, divergent_iters = self.MCMC_run(
            sigma0=sigma0, hyperparameters=hyperparameters
        )
        chains = np.asarray(chains)

        # Format chains
        self.chains = self._format_chains(chains, divergent_iters)
        return self.chains

    def set_stop_criterion(self, max_iterations=10000, r_hat=None):
        self._max_iters = max_iterations
        self._r_hat = r_hat

    def MCMC_run(
        self, save_point_like=None, sigma0=None, hyperparameters=None
    ):
        samplers = []

        # Apply a transformation (if given). From this point onward the MCMC
        # sampler will see only the transformed search space and will know
        # nothing about the model parameter space.
        if self._transform is not None:
            # Convert log pdf
            log_pdf = self._transform.convert_log_pdf(self._log_posterior)
            # Convert initial positions
            x0 = [self._transform.to_search(x) for x in self._initial_params]

            # Convert sigma0, if provided
            if sigma0 is not None:
                sigma0 = np.asarray(sigma0)
                n_parameters = log_pdf.n_parameters()
                # Make sure sigma0 is a (covariance) matrix
                if np.product(sigma0.shape) == n_parameters:
                    # Convert from 1d array
                    sigma0 = sigma0.reshape((n_parameters,))
                    sigma0 = np.diag(sigma0)
                elif sigma0.shape != (n_parameters, n_parameters):
                    # Check if 2d matrix of correct size
                    raise ValueError(
                        'sigma0 must be either a (d, d) matrix or a (d, ) '
                        'vector, where d is the number of parameters.')
                sigma0 = self._transform.convert_covariance_matrix(
                    sigma0, x0[0]
                )
        else:
            log_pdf = self._log_posterior
            x0 = self._initial_params

        for i in range(0, self._n_runs):
            point = x0[i]
            samplers.append(self._sampler(point, sigma0=sigma0))
            if samplers[-1].needs_initial_phase():
                samplers[-1].set_initial_phase(True)

        num_initial = 0.1*self._required_iters
        list_sample = [[]]*self._n_runs
        if save_point_like is not None:
            list_pointwise = [[]]*self._n_runs
            # log_likelihood = log_pdf.get_log_likelihood()
        
        final_iteration = self._max_iters
        timer = pints.Timer()

        print("iter", end=' ')
        for chain in range(len(samplers)):
            print('\t', "chain "+str(chain), end=' ')
        print('\t', "R_hat", '\t\t', "Time")

        i = 0
        r_hat = np.NaN

        # Create evaluator object
        f = log_pdf
        if samplers[0].needs_sensitivities():
            f = f.evaluateS1
        evaluator = pints.SequentialEvaluator(f)
        print_func = [None]*self._n_runs
        chain_lengths = [0]*self._n_runs
        n_return = 0

        while i < final_iteration:
            # update each chain and pointwise using ask/tell
            if (i == num_initial) & samplers[-1].needs_initial_phase():
                print("\n...........................")
                print("End of Initial Phase")
                print("...........................")
                for sampler in samplers:
                    sampler.set_initial_phase(False)
            
            theta_hat = [alg.ask() for alg in samplers]
            try:
                fxs = evaluator.evaluate(theta_hat)
                fxs_iterator = iter(fxs)
                for chain, alg in enumerate(samplers):
                    # lp = fxs(theta_hat)
                    reply = alg.tell(next(fxs_iterator))
                    if reply is not None:
                        theta_g, f_theta, accepted = reply
                        if self._transform:
                            model_theta_g = self._transform.to_model(theta_g)
                        else:
                            model_theta_g = theta_g
                        list_sample[chain] = list_sample[chain]+[model_theta_g]
                        chain_lengths[chain] += 1
                        if issubclass(
                            self._sampler,
                            (pints.HamiltonianMCMC, pints.NoUTurnMCMC)
                        ):
                            print_func[chain] = round(
                                chain_lengths[chain]/i, 4
                            )
                        else:
                            print_func[chain] = round(f_theta[0], 4)

                        # if save_point_like is not None:
                        #     if accepted:
                        #         list_pointwise[chain] = (
                        #             list_pointwise[chain] +
                        #             [log_likelihood.compute_pointwise_ll(
                        #                 model_theta_g
                        #             )]
                        #         )
                        #     else:
                        #         list_pointwise[chain] = (
                        #             list_pointwise[chain] +
                        #             [list_pointwise[chain][-1]]
                        #         )
            except:
                print("Error for iteration", i, ", parameters:", theta_hat)
                raise
            
            n_return = np.min(chain_lengths)
            if self._r_hat is not None:
                if i == num_initial + self._required_iters:
                    print("\n...........................")
                    print("Sart of Convergence Testing")
                    print("...........................")
                # Have the chains converged?
                test_r_hat = n_return > 0
                test_r_hat = test_r_hat & (i >= num_initial)
                test_r_hat = test_r_hat & (i % 100 == 0)
                if test_r_hat:
                    samples_test = [list[-n_return:] for list in list_sample]
                    r_hat = pints.rhat(
                        np.asarray(samples_test))
                    # [:, -self._required_iters:, :])
                    terminate = (n_return >= num_initial+self._required_iters)
                    terminate = terminate & np.all(r_hat <= self._r_hat)
                    if terminate:
                        final_iteration = i
                        print("complete")
            elif n_return >= num_initial+self._required_iters:
                final_iteration = i
                print("complete")

            i += 1
            if (i < 10) or (i % 100 == 0):
                print_string = str(i+1) + ' \t'
                for chain in range(0, self._n_runs):
                    print_string += str(print_func[chain])+'     \t'
                if self._r_hat is not None:
                    print_string += str(round(np.average(r_hat), 4))
                    print_string += '     \t'
                print_string += timer.format(timer.time())
                print(
                    "\r" + print_string,
                    sep=' ',
                    end='        ',
                    flush=True
                )

        if save_point_like is not None:
            np.savez_compressed(save_point_like, *list_pointwise)

        # If Hamiltonian Monte Carlo, get number of divergent
        # iterations # TODO: Implement
        divergent_iters = None
        if issubclass(
                self._sampler, (pints.HamiltonianMCMC, pints.NoUTurnMCMC)):
            divergent_iters = [
                s.divergent_iterations() for s in sampler.samplers()
            ]
        return_samples = [list[-n_return:] for list in list_sample]

        return return_samples, divergent_iters


class OptimisationController(chi.OptimisationController):
    """
    Sets up an optimisation routine that attempts to find the parameter values
    that maximise a :class:`pints.LogPosterior`. If multiple log-posteriors are
    provided, the posteriors are assumed to be structurally identical and only
    differ due to different data sources.

    By default the optimisation is run 5 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.

    :param log_posterior: The log-posterior from which is sampled.
    :type log_posterior: chi.LogPosterior, chi.HierarchicalLogPosterior
    :param seed: Seed for the random initialisation. Initial points are sampled
        from the log-prior.
    :type seed: int
    """

    def __init__(self, log_posterior, seed=None):
        super(OptimisationController, self).__init__(log_posterior, seed)

        # Set default optimiser
        self._optimiser = pints.CMAES

    def run(
            self, n_max_iterations=10000, show_run_progress_bar=False,
            log_to_screen=False
        ):
        """
        Runs the optimisation and returns the maximum a posteriori probability
        parameter estimates in from of a :class:`pandas.DataFrame` with the
        columns 'ID', 'Parameter', 'Estimate', 'Score' and 'Run'.

        The number of maximal iterations of the optimisation routine can be
        limited by setting ``n_max_iterations`` to a finite, non-negative
        integer value.

        Parameters
        ----------

        n_max_iterations
            The maximal number of optimisation iterations to find the MAP
            estimates for each log-posterior. By default the maximal number
            of iterations is set to 10000.
        show_run_progress_bar
            A boolean flag which indicates whether a progress bar for looping
            through the optimisation runs is displayed.
        log_to_screen
            A boolean flag which indicates whether the optimiser logging output
            is displayed.
        """

        # Initialise result dataframe
        result = pandas.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Time', 'Run'])

        # Initialise intermediate container for individual runs
        run_result = pandas.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Time', 'Run'])
        run_result['Parameter'] = self._log_posterior.get_parameter_names()

        # Set ID of individual (or IDs of parameters, if hierarchical)
        run_result['ID'] = self._log_posterior.get_id()

        # Run optimisation multiple times
        for run_id in tqdm(
                range(self._n_runs), disable=not show_run_progress_bar):
            opt = pints.OptimisationController(
                function=self._log_posterior,
                x0=self._initial_params[run_id, :],
                method=self._optimiser,
                transformation=self._transform)

            # Configure optimisation routine
            opt.set_log_to_screen(log_to_screen)
            opt.set_max_iterations(iterations=n_max_iterations)
            opt.set_parallel(self._parallel_evaluation)

            # Find optimal parameters
            try:
                estimates, score = opt.run()
            except Exception:
                # If inference breaks fill estimates with nan
                estimates = [np.nan] * self._log_posterior.n_parameters()
                score = np.nan

            # Save estimates and score of runs
            run_result['Estimate'] = estimates
            run_result['Score'] = score
            run_result['Run'] = run_id + 1
            run_result['Time'] = opt.time()
            result = pandas.concat([result, run_result])

        return result

    def make_initial_point(
        self, n_runs, population_parameters, individual_parameters=None
    ):
        n_top = self._log_posterior.n_parameters(exclude_bottom_level=True)
        n_bottom = self._log_posterior.n_parameters(exclude_bottom_level=False) - n_top
        log_likelihood = self._log_posterior.get_log_likelihood()

        population_parameters = np.atleast_2d(population_parameters)
        if population_parameters.shape[1] != n_top:
            raise ValueError(
                'population parameters must be same length as the ' +
                'log-posterior population parameters')

        population_model = log_likelihood.get_population_model()
        dims = []
        current_dim = 0
        
        if individual_parameters is not None:
            individual_parameters = np.atleast_2d(individual_parameters)
            if individual_parameters.shape[1]!=n_bottom:
                raise ValueError(
                    'individual parameters must match the ' +
                    'number of individuals')


        if isinstance(population_model, chi.ReducedPopulationModel):
            population_model = population_model.get_population_model()
        try:
            pop_models = population_model.get_population_models()
        except AttributeError:
            pop_models = [population_model]
        for pop_model in pop_models:
            n_dim = pop_model.n_dim()
            if isinstance(
                    pop_model, (chi.PooledModel, chi.HeterogeneousModel)):
                current_dim += n_dim
                continue
            end_dim = current_dim + n_dim
            dims += list(range(current_dim, end_dim))
            current_dim = end_dim

        initial_points = np.empty((n_runs, n_top+n_bottom))

        for i in range(0, n_runs):
            if individual_parameters is None:
                n_ids = log_likelihood.n_log_likelihoods()
                covariates = log_likelihood._covariates
                bottom_parameters = []
                population_model = log_likelihood.get_population_model()
                bottom_parameters.append(population_model.sample(
                    parameters=population_parameters[i, :],
                    n_samples=n_ids, covariates=covariates))
                # Remove pooled dimensions
                # (Pooled and heterogen. dimensions do not count as bottom parameters)
                for idx, bottom_params in enumerate(bottom_parameters):
                    bottom_parameters[idx] = bottom_params[:, dims].flatten()
                bottom_parameters = np.vstack(bottom_parameters)
            else:
                bottom_parameters = individual_parameters[i, :]

            # print(self._initial_params[run-1, n_bottom:])
            initial_points[i, n_bottom:] = population_parameters[i, :]
            initial_points[i, :n_bottom] = bottom_parameters
        return initial_points

    def set_initial_point(
        self, run_ids, population_parameters, individual_parameters=None
    ):
        """
        Sets the initial starting points for a selection of the runs.
        """
        # n_top = self._log_posterior.n_parameters(exclude_bottom_level=True)
        # n_bottom = self._log_posterior.n_parameters(exclude_bottom_level=False) - n_top
        if isinstance(run_ids, int):
            run_ids = [run_ids]
        run_ids = np.asarray(run_ids)
        if any(run_ids > self._n_runs) or any(run_ids <= 0):
            raise ValueError(
                'run_ids must be an int or list of int between 1 and the ' +
                'currently set number of runs. Use set_n_runs to increase ' +
                'number of runs set')

        initial_points = self.make_initial_point(len(run_ids), population_parameters, individual_parameters)
        for i, run in enumerate(run_ids):
            self._initial_params[run-1, :] = initial_points[i, :]


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
