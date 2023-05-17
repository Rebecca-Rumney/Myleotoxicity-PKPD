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
        self.chains = None

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
        self,
        n_iterations,
        hyperparameters=None,
        log_to_screen=False,
        reset=True
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
        :type n_iterations: int
        :param hyperparameters: A list of hyperparameters for the sampling
            method. If ``None`` the default hyperparameters are set.
        :type hyperparameters: list[float], optional
        :param log_to_screen: A boolean flag which can be used to print
            the progress of the runs to the screen. The progress is printed
            every 500 iterations.
        :type log_to_screen: bool, optional
        :param reset: A boolean flag which determines whether the chains are
            reset or the new samples are added to the existing chains.
        :type log_to_screen: bool, optional
        """

        if reset or (self.chains is None):
            initial_points = self._initial_params
        else:
            prev_params = self.chains.sel({'draw': max(self.chains.draw)})
            prev_pop_params = []
            prev_ind_params = []
            for var in prev_params.data_vars:
                last_sample = prev_params[var]
                if 'individual' in last_sample.dims:
                    prev_ind_params.append(last_sample.values)
                else:
                    prev_pop_params.append(last_sample.values)
            prev_pop_params = np.asarray(prev_pop_params).transpose()
            prev_ind_params = np.asarray(prev_ind_params).transpose(1, 2, 0)
            initial_points = self.make_initial_point(
                self._n_runs,
                prev_pop_params,
                prev_ind_params.reshape((self._n_runs, -1))
            )

        # Set up sampler
        # TODO: Allow R-hat to be incorporated
        sampler = pints.MCMCController(
            log_pdf=self._log_posterior,
            chains=self._n_runs,
            x0=initial_points,
            method=self._sampler,
            transformation=self._transform)

        # Configure sampling routine
        sampler.set_log_to_screen(log_to_screen)
        sampler.set_log_interval(iters=20, warm_up=3)
        sampler.set_max_iterations(iterations=n_iterations)
        sampler.set_parallel(self._parallel_evaluation)

        if hyperparameters:
            for s in sampler.samplers():
                s.set_hyper_parameters(hyperparameters)

        # Run sampling routine
        chains = sampler.run()

        # If Hamiltonian Monte Carlo, get number of divergent
        # iterations
        divergent_iters = None
        if issubclass(
                self._sampler, (pints.HamiltonianMCMC, pints.NoUTurnMCMC)):
            divergent_iters = [
                s.divergent_iterations() for s in sampler.samplers()]

        # Format chains
        if reset:
            self.reset_chains()
        else:
            chains = chains[:, int(n_iterations*0.1):]
        self.add_samples(self._format_chains(chains, divergent_iters))
        return self.chains

    def reset_chains(self):
        self.chains = None

    def add_samples(self, new_chains):
        if self.chains is None:
            self.chains = new_chains
        else:
            new_chains = new_chains.assign_coords(
                draw=(new_chains.draw + len(self.chains.draw))
            )
            self.chains = xr.concat((
                self.chains,
                new_chains
            ), dim='draw')

    def set_stop_criterion(self, max_iterations=10000, r_hat=None):
        self._max_iters = max_iterations
        self._r_hat = r_hat


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
        self,
        n_max_iterations=10000,
        show_run_progress_bar=False,
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
        post = self._log_posterior
        n_top = post.n_parameters(exclude_bottom_level=True)
        n_bottom = post.n_parameters(exclude_bottom_level=False) - n_top
        log_likelihood = post.get_log_likelihood()

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
                # Remove pooled dimensions (Pooled and heterogen. dimensions
                # do not count as bottom parameters)
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


if __name__ == "__main__":
    pass
