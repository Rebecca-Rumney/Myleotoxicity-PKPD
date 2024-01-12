import numpy as np
import chi
import pints
import plotly.express.colors as pxclrs
import plotly.colors as pclrs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import arviz as az
import arviz.labels as azl
from cycler import cycler
import myokit
from scipy.optimize import minimize, brentq
from scipy.interpolate import interp1d, make_interp_spline, PPoly
from scipy.interpolate import UnivariateSpline, PchipInterpolator


class ProfileLogLikelihood():

    def __init__(self, log_likelihood, params_ref, opts) -> None:
        self.set_options(opts, reset=True)
        self.slice_func = self.create_func(
            log_likelihood, params_ref, profile=False
        )
        if self.opts['slice']:
            self.f = self.slice_func
        else:
            self.f = self.create_func(
                log_likelihood, params_ref, profile=True
            )
        self.MLE = params_ref.copy()
        self.ref_score = log_likelihood(params_ref)
        self.l_star = self.ref_score - self.opts['alpha']
        self.param_range = {}
        self.result = {}
        self.pints_optimisers = dict(zip(
            [cls.__name__ for cls in pints.Optimiser.__subclasses__()],
            pints.Optimiser.__subclasses__()
        ))

    def run(self, i_param, n_points=100, opts=None):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)
        if i_param not in self.param_range.keys():
            self.set_param_range(i_param, adapt=True)

        # Reset results
        self.result[i_param] = {}

        if local_opts['method'] == 'quadratic approx.':
            self.ci_poly_approx(i_param, opts=local_opts)
            profile_ll = self.ll_from_ci(i_param, opts=local_opts)
        elif local_opts['method'] == 'piecewise calc.':
            profile_ll = self.ll_piecewise(i_param, opts=local_opts)
            self.ci_from_ll(i_param, opts=local_opts)
        elif local_opts['method'] == 'sequential calc.':
            profile_ll = self.ll_sequential(i_param, opts=local_opts)
            self.ci_from_ll(i_param, opts=local_opts)
        else:
            raise NameError(
                'Unknown profile likelihood method'
            )

        # Get points
        param_range = self.param_range[i_param][0]
        x_values = np.linspace(
            param_range[0], param_range[1], n_points
        )
        ll_values = profile_ll(x_values)

        return np.array([x_values, ll_values])

    def optimise(
            self, func, start, fix=None, minimise=False, transformation=None, opts=None
        ):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)

        # PINTS optimisers minimise.
        # To maximise the function, f, we will need to minimise -f
        if minimise:
            sign = 1
        else:
            sign = -1

        # Transform start point into [eta, pop]
        opt_start = start.copy()
        if transformation is not None:
            opt_start = transformation.to_search(opt_start)

        if fix is not None:
            fix = np.asarray(fix)
            # If there are fixed params, determine what values need to be deleted and inserted
            if len(fix.shape) == 1:
                delete_arg = int(fix[0])
                insert_arg = int(fix[0])
                fix_values = fix[1]
            else:
                fix = fix[:, np.argsort(fix[0])]
                insert_arg = fix[0].astype(int)-np.arange(len(fix[0]))
                delete_arg = fix[0].astype(int)
                fix_values = fix[1]

            # Delete fixed values from the starting point
            opt_start = np.delete(opt_start, delete_arg)

            # Create the function to minimise
            def minimise_func(reduced_params):
                # insert the fixed values to the correct spots
                full_params = np.insert(reduced_params, insert_arg, fix_values)
                # transform parameters back from [eta, pop] to [ind, pop]
                if transformation is not None:
                    full_params = transformation.to_model(full_params)
                return sign*func(full_params)
        else:
            def minimise_func(full_params):
                if transformation is not None:
                    full_params = transformation.to_model(full_params)
                return sign*func(full_params)
        max_iter = local_opts['maxiter']
        max_unchanged = local_opts['max_unchanged']
        opt_package = local_opts['opt_package']
        opt_method = local_opts['optimiser']
        if opt_package == "pints":
            if opt_method not in self.pints_optimisers.keys():
                raise NameError(
                    "pints does not have optimiser " + opt_method + ". " +
                    "Perhaps you need to change 'opt_package' option to " +
                    "'scipy'."
                )
            else:
                optimiser = self.pints_optimisers['opt_method']
            xbest, fbest = pints.fmin(
                minimise_func, opt_start, max_iter=max_iter,
                max_unchanged=max_unchanged, method=optimiser
            )
        elif opt_package == "scipy":
            options = {}
            if max_iter is not None:
                options['maxiter'] = max_iter
            result = minimize(
                minimise_func, opt_start, method=opt_method, options=options
            )
            xbest = result.x
            fbest = result.fun

        if fix is not None:
            xbest = np.insert(xbest, insert_arg, fix_values)
        if transformation is not None:
            xbest = transformation.to_model(xbest)

        return xbest, sign*fbest

    def ll_sequential(self, i_param, opts=None):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)

        param_range = self.param_range[i_param][0]
        x_rec = np.linspace(param_range[0], param_range[1], local_opts['max_N_opts'])

        # Start from the centre, for more accurate extrapolating
        lower = x_rec <= self.MLE[i_param]
        x_lower = x_rec[lower][::-1]
        x_upper = x_rec[np.logical_not(lower)]
        # x_rec = np.concatenate((x_lower, x_upper))
        ll_rec = [np.empty_like(x_lower), np.empty_like(x_upper)]
        param_rec = [
            np.empty((len(x_lower), len(self.MLE))),
            np.empty((len(x_upper), len(self.MLE)))
        ]
        proj_rec = param_rec.copy()

        LU = 0
        for x_LU in (x_lower, x_upper):
            for i_x, x in enumerate(x_LU):
                if i_x <= 1:
                    # Set current approximation of optimum parameters to MLE
                    curr = self.MLE.copy()
                elif local_opts['interp'] > 0:
                    if i_x > local_opts['interp']+1:
                        k = local_opts['interp']
                    else:
                        k = (i_x-1)

                    x_interp = x_LU[i_x-1-k: i_x]
                    param_interp = param_rec[LU][i_x-1-k: i_x]

                    if LU == 0:
                        x_interp = x_interp[::-1]
                        param_interp = param_interp[::-1]
                    interpolator = make_interp_spline(
                        x_interp,
                        param_interp,
                        k=k, axis=0
                    )
                    curr = interpolator(x)

                else:
                    curr = param_rec[LU][i_x-1]
                proj_rec[LU][i_x] = curr
                param_rec[LU][i_x], ll_rec[LU][i_x] = self.f(
                    x, i_param, curr=curr
                )
            LU = 1

        x_rec = np.concatenate((x_lower[::-1], x_upper))
        ll_rec = np.concatenate((ll_rec[0][::-1], ll_rec[1]))
        param_rec = np.concatenate((param_rec[0][::-1], param_rec[1]))
        proj_rec = np.concatenate((proj_rec[0][::-1], proj_rec[1]))

        if local_opts['normalise']:
            ll_rec = ll_rec - self.l_star  # Normalise the result
        param_interpolator = PchipInterpolator(
                x_rec, param_rec, axis=0
            )
        ll_interpolator = PchipInterpolator(
                x_rec, ll_rec
            )

        result = self.result[i_param]
        result.update({
            'opt points': (x_rec, ll_rec, param_rec),
            'projected optimum': proj_rec,
            'profile ll': ll_interpolator,
            'profile param': param_interpolator
        })
        self.result[i_param] = result

        return ll_interpolator

    def ll_piecewise(self, i_param, opts=None):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)

        param_range = self.param_range[i_param][0]
        x_rec = [param_range[0], self.MLE[i_param], param_range[1]]
        param_rec = [None, self.MLE, None]
        ll_rec = [None, self.ref_score, None]
        param_rec[0], ll_rec[0] = self.f(x_rec[0], i_param, curr=self.MLE)
        param_rec[2], ll_rec[2] = self.f(x_rec[2], i_param, curr=self.MLE)

        proj_rec = [self.MLE, self.MLE, self.MLE]
        stop_criteria = [False]*len(x_rec)
        n_del = 1

        while len(x_rec) <= self.opts['max_N_opts']:
            x_insert = []
            insert_args = []
            ll_insert = []
            param_insert = []

            # Determine where to add new points
            if x_rec[0] != param_range[0]:
                insert_args += [0, 0]
                x_insert.append(param_range[0])
                x_insert.append((param_range[0]+x_rec[0])*0.5)
            for i in range(1, len(x_rec)):
                skip = stop_criteria[i-1] and stop_criteria[i]
                if len(x_rec) > self.opts['min_N_opts']+1 and skip:
                    continue
                else:
                    insert_args += [i]
                    x_insert.append((x_rec[i-1] + x_rec[i])*0.5)
            if x_rec[-1] != param_range[1]:
                insert_args += [len(x_rec), len(x_rec)]
                x_insert.append((param_range[1]+x_rec[-1])*0.5)
                x_insert.append(param_range[1])
            x_insert = np.asarray(x_insert)

            # Create intitial prediction using interpolation
            interpolator = make_interp_spline(
                    x_rec, param_rec,
                    k=local_opts['interp'], axis=0
                )
            proj_values = interpolator(x_insert)
            proj_rec = np.insert(proj_rec, insert_args, proj_values)

            # Maximise other params and Generate profile ll score
            for i, x in enumerate(x_insert):
                param_value, ll = self.f(x, i_param, curr=proj_values[i])
                ll_insert.append(ll)
                param_insert.append(param_value)

            arg_delete = None
            if len(x_rec) >= local_opts['min_N_opts']:
                # Whether the approximation is good
                interpolator = PchipInterpolator(
                        x_insert, param_insert,
                        k=local_opts['interp'], axis=0
                    )
                error = np.sum(np.abs(interpolator(x_rec) - param_rec), axis=1)
                error_new = np.sum(
                    np.abs(proj_values - np.asarray(param_insert)), axis=1
                )
                stop_criteria = error <= 1e-2
                stop_criteria = np.insert(
                    stop_criteria, insert_args, error_new <= 1e-2
                )

                # Find outliers
                arg_delete = np.argpartition(error, len(error)-n_del)[-n_del:]
                arg_delete = arg_delete[
                    np.logical_not(stop_criteria[arg_delete])
                ]
                arg_delete += np.count_nonzero(np.tile(
                    insert_args, (n_del, 1)
                ).transpose() <= arg_delete, axis=0)

            else:
                stop_criteria += [False]*len(x_insert)

            # Insert the new values into arrays
            x_rec = np.insert(x_rec, insert_args, x_insert)
            ll_rec = np.insert(ll_rec, insert_args, ll_insert)
            param_rec = np.insert(param_rec, insert_args, param_insert, axis=0)

            if arg_delete is not None:
                # Delete Outliers
                x_rec = np.delete(x_rec, arg_delete)
                ll_rec = np.delete(ll_rec, arg_delete)
                param_rec = np.delete(param_rec, arg_delete, axis=0)
                proj_rec = np.delete(proj_rec, arg_delete, axis=0)
                stop_criteria = np.delete(stop_criteria, arg_delete)

            if np.all(stop_criteria):
                break

        if local_opts['normalise']:
            ll_rec = ll_rec - self.l_star  # Normalise the result
        param_interpolator = PchipInterpolator(
                x_rec, param_rec, k=local_opts['interp'], axis=0
            )
        ll_interpolator = PchipInterpolator(
                x_rec, ll_rec, k=local_opts['interp']
            )

        result = self.result[i_param]
        result.update({
            'opt points': (x_rec, ll_rec, param_rec),
            'projected optimum': proj_rec,
            'profile ll': ll_interpolator,
            'profile param': param_interpolator
        })
        self.result[i_param] = result

        return ll_interpolator

    def ll_from_ci(self, i_param, CI=None, opts=None):
        # Update options
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)

        # Get relevant current result
        result = self.result[i_param]

        # Get Confidence intervals
        if CI is None:
            x_CI, ll_CI, params_CI = result['CI']
        else:
            x_CI, ll_CI, params_CI = CI

        # If ci was calculated from this class, find points of optimisation
        if 'opt points' in result.keys() and CI is None:
            x_rec, ll_rec, param_rec = result['opt points']
        else:
            x_rec = x_CI.copy()
            ll_rec = ll_CI.copy()
            param_rec = params_CI.copy()

        param_range = self.param_range[i_param][0]
        if local_opts['approx shape N'] > 0:
            x_shape_points = np.linspace(
                param_range[0], param_range[1], local_opts['approx shape N']
            )
            param_interpolator = make_interp_spline(
                    x_CI, params_CI, k=local_opts['interp'], axis=0
                )
            for x in x_shape_points:
                param, ll = self.f(x, i_param, curr=param_interpolator(x))
                param_rec = np.append(param_rec, [param], axis=0)
                x_rec = np.append(x_rec, x)
                if local_opts['normalise']:
                    ll = ll - self.l_star
                ll_rec = np.append(ll_rec, ll)
            sort_args = np.argsort(x_rec)
            x_rec = x_rec[sort_args]
            ll_rec = ll_rec[sort_args]
            param_rec = param_rec[sort_args]
            result['opt points'] = (x_rec, ll_rec, param_rec)

        weights = np.ones(len(x_rec))
        weights[np.isin(x_rec, x_CI)] = 80
        weights[x_rec == self.MLE[i_param]] = 200
        inf_result = np.logical_not(np.isfinite(ll_rec))
        if np.any(inf_result):
            ll_rec[inf_result] = 2
            weights[inf_result] = 0
        ll_interpolator = UnivariateSpline(
                x_rec, ll_rec,
                w=weights, s=1e-2*len(weights)
            )
        # ll_interpolator = PchipInterpolator(x_rec, ll_rec, extrapolate=True)
        param_interpolator = PchipInterpolator(
                x_rec, param_rec, axis=0, extrapolate=True
            )

        result['profile ll'] = ll_interpolator
        result['profile param'] = param_interpolator

        self.result[i_param] = result

        return ll_interpolator

    def ci_from_ll(self, i_param, opts=None):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)
        result = self.result[i_param]

        ll_interpolator = result['profile ll']
        param_interpolator = result['profile param']
        param_range = self.param_range[i_param][0]
        if local_opts['normalise']:
            roots = np.sort(ll_interpolator.roots(extrapolate=False))
        else:
            roots = np.sort(
                ll_interpolator.solve(y=self.l_star, extrapolate=False)
            )

        x_CI = [param_range[0], self.MLE[i_param], param_range[1]]
        if len(roots) <= 2:
            identifiable_CI = ['unident', 'unident']
            for x in roots:
                if x < self.MLE[i_param]:
                    x_CI[0] = x
                    identifiable_CI[0] = 'ident'
                else:
                    x_CI[2] = x
                    identifiable_CI[1] = 'ident'
        else:
            identifiable_CI = [None, None]
        ll_CI = ll_interpolator(x_CI)
        param_CI = param_interpolator(x_CI)

        result['CI'] = (x_CI, ll_CI, param_CI)
        result['identifiabilty'] = identifiable_CI
        self.result[i_param] = result
        return x_CI, ll_CI, param_CI

    def ci_poly_approx(self, i_param, opts=None):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)
        result = self.result[i_param]
        if local_opts['interp'] < 2:
            local_opts['interp'] = 2

        # First step: Estimate CI using slice function
        param_range = self.param_range[i_param][0]
        x_CI = [param_range[0], self.MLE[i_param], param_range[1]]
        param_CI = [None, self.MLE, None]
        ll_CI = [None, self.ref_score, None]

        def solve_slice(x, L_U):
            param_CI[L_U+1], ll = self.slice_func(x, i_param)
            return ll - self.l_star
        x_CI[0] = brentq(solve_slice, x_CI[0], x_CI[1], args=(-1))
        x_CI[2] = brentq(solve_slice, x_CI[1], x_CI[2], args=(1))

        # Optimise at this point to find the log-likelihood
        param_CI[0], ll_CI[0] = self.f(x_CI[0], i_param, curr=param_CI[0])
        param_CI[2], ll_CI[2] = self.f(x_CI[2], i_param, curr=param_CI[2])

        x_rec = x_CI.copy()
        ll_rec = ll_CI.copy()
        param_rec = param_CI.copy()
        proj_rec = [self.MLE, self.MLE, self.MLE]

        identifiable_CI = [None]*2
        unchanged_iters_L = 0
        unchanged_iters_U = 0

        # Iteratively estimate CI approximating space as polynomial
        while len(x_rec) <= self.opts['max_N_opts']:
            # Solve LL(x)-L_star=0 assuming polynomial
            poly_spline = PPoly.from_spline(make_interp_spline(
                x_CI, ll_CI - self.l_star,
                k=local_opts['interp'], axis=0
            ))
            roots = np.sort(poly_spline.roots())

            # Approximate maximum parameters.
            interpolator = make_interp_spline(
                    x_CI, param_CI,
                    k=local_opts['interp'], axis=0
                )
            if identifiable_CI[0] is None:
                approx_param = interpolator(roots[0])
                # Calculate true max params and likelihood at the lower root
                param_CI[0], ll_CI[0] = self.f(
                    roots[0], i_param, curr=approx_param
                )
                x_CI[0] = roots[0]

                # Record these values
                param_rec = np.insert(param_rec, 0, param_CI[0], axis=0)
                proj_rec = np.insert(proj_rec, 0, approx_param, axis=0)
                ll_rec = np.insert(ll_rec, 0, ll_CI[0])
                x_rec = np.insert(x_rec, 0, x_CI[0])

                # Check termination status
                if np.abs(ll_CI[0]-self.l_star) < local_opts['T_ident']:
                    identifiable_CI[0] = 'Ident'
                elif np.abs(ll_rec[0] - ll_rec[1]) < local_opts['T_unident']:
                    unchanged_iters_L += 1
                    if unchanged_iters_L >= 5:
                        identifiable_CI[0] = 'Unident'
                else:
                    unchanged_iters_L = 0

            if identifiable_CI[1] is None:
                approx_param = interpolator(roots[0])
                # Calculate true max params and likelihood at the upper root
                param_CI[2], ll_CI[2] = self.f(
                    roots[1], i_param, curr=approx_param
                )
                x_CI[2] = roots[1]

                # Record these values
                param_rec = np.insert(
                    param_rec, len(x_rec), param_CI[2], axis=0
                )
                proj_rec = np.insert(
                    proj_rec, len(x_rec),  approx_param, axis=0
                )
                ll_rec = np.insert(ll_rec, len(x_rec), ll_CI[2])
                x_rec = np.insert(x_rec, len(x_rec), roots[1])

                # Check termination status
                if np.abs(ll_CI[2]-self.l_star) < local_opts['T_ident']:
                    identifiable_CI[1] = 'Ident'
                elif np.abs(ll_rec[-1] - ll_rec[-2]) < local_opts['T_unident']:
                    unchanged_iters_U += 1
                    if unchanged_iters_U >= local_opts['unchanged_opts']:
                        identifiable_CI[1] = 'Unident'
                else:
                    unchanged_iters_U = 0

            if all(identifiable_CI):
                break

        if local_opts['normalise']:  # Normalise the result
            ll_rec = ll_rec - self.l_star
            ll_CI = ll_CI - self.l_star

        # Sort the results for interpolation
        sort = np.argsort(x_rec)
        x_rec = x_rec[sort]
        ll_rec = ll_rec[sort]
        param_rec = param_rec[sort]

        result.update({
            'CI': (x_CI, ll_CI, param_CI),
            'opt points': (x_rec, ll_rec, param_rec),
            'identifiabilty': identifiable_CI,
            'projected optimum': proj_rec
        })
        self.result[i_param] = result
        return x_CI, ll_CI, param_CI

    def set_options(self, opts, reset=False):
        if reset:
            self.opts = {
                'optimiser': 'Powell',
                'opt_package': 'scipy',
                'method': 'quadratic approx.',
                'minimise': False,
                'transformation': None,
                'normalise': True,
                'slice': False,
                'alpha': 1.92,
                'view_aim': -3*1.92,
                'interp': 2,
                'T_ident': 1e-3,
                'T_unident': 1e-4,
                'max_N_opts': 50,
                'min_N_opts': 10,
                'unchanged_opts': 5,
                'approx shape N': 10,
                'maxiter': None,
                'max_unchanged': (200, 1e-11)
            }
        self.opts.update(opts)

    def create_func(self, function, params_ref, profile=True):
        if profile:
            def profile_function(param_value, param_arg, curr):
                opt_param, score = self.optimise(
                    function,
                    curr,
                    fix=[param_arg, param_value]
                )
                return opt_param, score
            f = profile_function
        else:
            def slice_function(param_value, param_arg, curr=None):
                slice_param = params_ref.copy()
                slice_param[param_arg] = param_value
                score = function(slice_param)
                return slice_param, score
            f = slice_function
        return f

    def max_likelihood_estimate(self, n_runs=1, param_ref=None):
        """
        Finds and sets the maximum likelihood estimate (MLE).

        Parameters
        ----------

        n_runs
            The number of runs of the optimisation to determine the MLE. All
            results are retuned but only the best result is saved as the MLE.

        param_ref
            The starting point of the optisation. NDArray of shape (n_params, )
            or (n_runs, n_params). If none is given, the reference params
            initialised will be used
        """
        # return f
        pass  # Set up param range

    def set_param_range(self, i_param, bounds=None, adapt=False):
        """
        Sets the range in which profile likelihood is calculated.

        Parameters
        ----------

        i_param
            The parameter or list of parameters, theta_i, for which the
            param_range is calculated for.
        bounds
            The bounds of the parameter range. NDArray of shape (2, ) or
            (len(i_param), 2). If None is given or either bound is None,
            0.5*MLE_i and 1.5*MLE_i will be used.
        adapt
            Whether the bounds of the range should be shrunk to better view
            the loglikelihood curve around MLE. Aims to view between ll(MLE)
            and ll(MLE)-3*1.92. Can be changed in set_opts.
        """
        if bounds is None:
            bounds = np.full((len(i_param), 2), None)

        bounds = np.array(bounds)
        if bounds.ndim == 1:
            if len(bounds) > 2:
                raise TypeError()
            elif len(bounds) == 1:
                if bounds[0] is None:
                    bounds = np.full((len(i_param), 2), None)
                else:
                    raise TypeError(
                        "Bounds must be of shape (2, ), or (len(i_param), 2)"
                    )
            else:
                bounds = np.array([
                    [bounds[0]]*len(i_param),
                    [bounds[1]]*len(i_param)
                ])
                bounds = np.array(bounds).transpose()
        elif bounds.shape != (len(i_param), 2):
            raise TypeError(
                "Bounds must be of shape (2, ), or (len(i_param), 2)"
            )
        none_L = bounds[:, 0] == None
        none_U = bounds[:, 1] == None
        bounds[none_L, 0] = 0.5*(self.MLE[i_param])[none_L]
        bounds[none_U, 1] = 1.5*(self.MLE[i_param])[none_U]

        if isinstance(adapt, bool):
            adapt = [adapt]*2
        for j_param, bound in enumerate(bounds):
            x_min = bound[0]
            i_check = 0
            j_check = 0
            if adapt[0]:
                while i_check == 0 and j_check <= 20:
                    x_check = np.linspace(
                        self.MLE[i_param[j_param]], x_min, 10
                    )[1:]
                    for x in x_check:
                        _, score = self.slice_func(x, i_param[j_param])
                        if score - self.l_star < self.opts['view_aim']:
                            x_min = x_check[i_check]
                            break
                        i_check += 1
                    j_check += 1
            x_max = bound[1]
            i_check = 0
            j_check = 0
            if adapt[1]:
                while i_check == 0 and j_check <= 20:
                    x_check = np.linspace(
                        self.MLE[i_param[j_param]], x_max, 10
                    )[1:]
                    for x in x_check:
                        _, score = self.slice_func(x, i_param[j_param])
                        if score-self.l_star < self.opts['view_aim']:
                            x_max = x_check[i_check]
                            break
                        i_check += 1
                    j_check += 1
            bounds[j_param] = [x_min, x_max]

        self.param_range.update(
            dict(zip(i_param, zip(bounds, [adapt]*len(i_param))))
        )


class Plot_Models():

    def __init__(
            self, pop_model=None, error_models=None, mech_model=None,
            prior_model=None, data=None
    ) -> None:
        # TODO: Create function for setting labels
        self.time_label = "Time, hours"
        self.PK_y_label = "Drug concentration, mg/L"
        self.PD_y_label = "Blood cell concentration, 10^3/microL"
        self.dose_unit = 'mg/Kg'
        self.dose_group_label = 'Dose, ' + self.dose_unit

        self.data_set = False
        # Initilise the default colour scheme
        self.default_colour = {
            "base": "rebeccapurple", "individual": "viridis",
            "zero dose": "turbo", "heat": "dense"
        }
        self.set_colour(self.default_colour)
        # Set the data
        if data is None:
            self.n_ind = 0
        else:
            self.set_data(data)
            self.n_ind = np.sum(self.n_ids_per_dose)

        # Set the non-pop models
        self.mech_model = mech_model
        self.error_model = error_models
        self.prior_model = prior_model

        # Set the pop model and get the mixed-effects/individual parameters
        self.pop_model = pop_model
        if self.pop_model is None:
            self.n_ind_params = 0
            self.ME_param_args = []
        else:
            if not self.data_set:
                self.n_ind = self.pop_model.n_ids()
            self.n_ind_params = self.n_ind*self.pop_model.n_hierarchical_dim()

            # Find the M-E parameters
            sp_dims = self.pop_model.get_special_dims()[0]
            n_params = self.pop_model.n_parameters()
            non_mix_params = np.asarray(
                [range(x, y) for _, _, x, y, _ in sp_dims]
            ).flatten()
            self.ME_param_args = [
                x + self.n_ind_params for x in range(0, n_params)
                if x not in non_mix_params
            ]

    def set_colour(self, colour_scheme):
        """
        Sets the colour schemes for plots.

        Parameters
        ----------

        colour_scheme
            Dict with labels as keys and plotly compatible colours or colour
            scales as values. The following labels may be supplied:
                - "base": Determines the colour used for most graphs where
                individuals are not plotted. Single plotly compatible colour
                must be provided. Defaults to css colour rebeccapurple.
                - "individual": Determines the colours used for graphs where
                individuals are compared. A plotly colour scale must be
                provided. Colour for each indvidual is selected from the scale,
                grouping individuals in the same dose group. Defaults to
                viridis.
                - "zero dose": Determines the colours used for graphs where
                individuals in the control group are compared. A plotly colour
                scale must be provided. If this scale is the same as
                "individual" then it will act as another dose group when
                selecting from "individual", otherwise it will ignore these
                control group individuals when selecting others from
                "individual" and instead select colours from "zero dose".
                Defaults to "turbo".
                - "heat": Determines the colours used for heat maps
                (unidirectional). A plotly colour scale must be provided.
                Defaults to "Dense".
        """

        if colour_scheme is None:
            colour_scheme = self.default_colour

        if "base" in colour_scheme.keys():
            self.base_colour = colour_scheme["base"]
            if self.base_colour is None:
                self.base_colour = self.default_colour["base"]
        if "individual" in colour_scheme.keys():
            self.ind_colour_scale = colour_scheme["individual"]
            if self.ind_colour_scale is None:
                self.ind_colour_scale = self.default_colour["individual"]
        if "zero dose" in colour_scheme.keys():
            self.ind_0_colour_scale = colour_scheme["zero dose"]
            if self.ind_0_colour_scale is None:
                self.ind_0_colour_scale = self.default_colour["zero dose"]
        if "heat" in colour_scheme.keys():
            self.heat_colour_scale = colour_scheme["heat"]
            if self.heat_colour_scale is None:
                self.heat_colour_scale = self.default_colour["zero dose"]

        if self.data_set:
            # Set up the colours for individuals
            diff_0_colour = self.ind_0_colour_scale != self.ind_colour_scale
            if (0.0 in self.dose_groups) and diff_0_colour:
                cols = [pxclrs.sample_colorscale(
                    pxclrs.get_colorscale(self.ind_0_colour_scale),
                    int(self.n_ids_per_dose.at[0.0]),
                    low=0.8,
                    high=1.0)
                ]
                n_drug_groups = len(self.dose_groups)-1
            else:
                cols = []
                n_drug_groups = len(self.dose_groups)

            for i, group in enumerate(self.dose_groups[-n_drug_groups:]):
                col_low = 1.5*i/(n_drug_groups*1.5-0.5)
                col_high = (1.5*i+1)/(n_drug_groups*1.5-0.5)
                cols.append(pxclrs.sample_colorscale(
                    pxclrs.get_colorscale(self.ind_colour_scale),
                    int(self.n_ids_per_dose.at[group]),
                    low=col_low,
                    high=col_high
                ))
            self.ind_colours = np.asarray(cols)

    def set_data(self, df):
        """
        Sets the data to plot.

        Parameters
        ----------

        df
            Pandas data frame containing the data.
        """
        dose_info = df[~np.isnan(df["Dose"])].groupby(["ID"])
        dose_amts = dose_info["Dose"]
        self.dose_times = dose_info["Time"].first()
        self.dose_amts = dose_amts.mean()
        self.dose_groups = df["Dose group"].unique()
        self.n_ids_per_dose = df.groupby("Dose group")["ID"].nunique()

        # Check whether data includes PD data.
        if 'circulating.R' in df['Observable'].unique():
            self.has_PD = True
            df_PK_graph = df.loc[
                df['Observable'] == 'PK_central.drug_concentration'
            ]

        else:
            self.has_PD = False
            df_PK_graph = df.loc[
                df['Observable'] == 'central.drug_concentration'
            ]

        # Reformat PK data
        df_PK_graph = df_PK_graph.rename(columns={
            'Dose group': self.dose_group_label,
            'Time': self.time_label,
            'Value': self.PK_y_label
        }, errors="raise")
        self.df_PK_graph = df_PK_graph.astype(
            {self.dose_group_label: 'category'}, errors="raise"
        )

        # Reformat PD data (empty if no PD data provided)
        df_PD_graph = df.loc[df['Observable'] == 'circulating.R']
        df_PD_graph = df_PD_graph.rename(columns={
            'Dose group': self.dose_group_label,
            'Time': self.time_label,
            'Value': self.PD_y_label
        }, errors="raise")
        self.df_PD_graph = df_PD_graph.astype(
            {self.dose_group_label: 'category'}, errors="raise"
        )

        self.data_set = True
        # Set up the colours for each individual
        self.set_colour({})

    def plot_prior(
            self, pop_params=None, individual_parameters=None,
            param_names=None, bounds=(None, None)
    ):
        # TODO: introduce option for histogram vs. function
        plot = self.plot_param_dist(
            self.prior_model,
            pop_params=pop_params,
            individual_parameters=individual_parameters,
            param_names=param_names,
            bounds=bounds
        )
        return plot

    def plot_pop_distribution(
            self, pop_params, individual_parameters=None, param_names=None
    ):
        plot = self.plot_param_dist(
            self.pop_model,
            pop_params=pop_params,
            individual_parameters=individual_parameters,
            param_names=param_names
        )
        return plot

    def plot_param_dist(
            self, distribution_model, pop_params=None,
            individual_parameters=None, param_names=None, bounds=(None, None),
            n_samples=50000
    ):
        """
        Samples a probability distribution from either a chi.PopulationModel
        or a pints.LogPrior.

        Parameters
        ----------

        distribution_model
            The model to sample from, must be an instance of either a
            chi.PopulationModel, pints.LogPrior class or a list of these.
        pop_params
            The parameters of the population model. Must be included if
            distribution model is an instance of a chi.PopulationModelas it is
            used to determine the parameter distribution to sample from. If
            included when the distribution model is an instance of a
            pints.LogPrior class, then the pop params will be plotted but will
            not affect the sampling.
        individual_parameters
            Dictionary of parameter names in the distribution to list of
            Individual parameters to plot. Does not affect sampling. If data
            has been set, length of each list of Individual_parameters must
            equal number of individuals in dataset.
        n_samples
            Number of samples of each parameter to draw for plotting.
        param_names
            Parameter names to use for subplot titles and the keys in
            individual_parameters. If not provided, names will be determined
            from the population model, or "Parameter 1", "Parameter 2", etc.
            is used for the log priors.

        Returns
        ----------
        Plotly figure with histograms of samples from the distribution of each
        parameter in the model.
        """

        # Visualise population model (parameter space)
        n_params = distribution_model.n_parameters()
        fig = make_subplots(rows=int((n_params+2)/3), cols=min(n_params, 3))
        plot_num = 0
        # pop_params = pop_params.copy()
        if isinstance(distribution_model, chi.ComposedPopulationModel):
            models = distribution_model.get_population_models()
        elif isinstance(distribution_model, list):
            models = distribution_model
        else:
            models = [distribution_model]

        dist_colour = 'lightgrey'
        if individual_parameters is None:
            individual_parameters = {}
            dist_colour = self.base_colour

        if self.data_set:
            ind_colour_selection = self.ind_colours
        else:
            ind_colour_selection = [self.base_colour]*max(
                [len(n) for n in individual_parameters.values()], default=0
            )

        for model in models:
            if isinstance(model, chi.LogNormalModel):
                n_dim = model.n_dim()
                default_names = model.get_dim_names()
                log_typs = pop_params[:n_dim]
                omegas = pop_params[n_dim:2*n_dim]
                const = -np.log(omegas)-0.5*np.log(2*np.pi)
                typs_height = np.exp(const-log_typs)
                pop_params = np.delete(pop_params, range(0, 2*n_dim))
                more_sampled = model.sample(
                    parameters=np.concatenate((log_typs, omegas)),
                    n_samples=n_samples
                )
                typs = np.exp(log_typs)
            elif isinstance(model, chi.PopulationModel):
                n_dim = model.n_dim()
                default_names = model.get_dim_names()
                typs = pop_params[:n_dim]
                typs_height = [1e10]*n_dim
                omegas = pop_params[n_dim:n_dim]
                pop_params = np.delete(pop_params, range(0, n_dim))
                more_sampled = model.sample(
                    parameters=typs,
                    n_samples=n_samples
                )
            elif isinstance(model, pints.LogPrior):
                n_dim = model.n_parameters()
                default_names = np.char.add(
                    ['Parameter ']*n_dim,
                    np.arange(plot_num, plot_num+n_dim).astype(str)
                )
                more_sampled = model.sample(
                    n=n_samples
                )
                if pop_params is not None:
                    typs = pop_params[:n_dim]
                    typs_height = model.cdf(typs)
                    pop_params = np.delete(pop_params, range(0, n_dim))

            for dim_num in range(0, n_dim):
                if param_names is None:
                    param = default_names[dim_num]
                else:
                    param = param_names[plot_num]
                row = int(plot_num/3)+1
                col = plot_num % 3 + 1

                if bounds[0] is None:
                    min_samp = more_sampled[:, dim_num].min()
                else:
                    min_samp = (bounds[0])[plot_num]
                if bounds[1] is None:
                    max_samp = more_sampled[:, dim_num].max()
                else:
                    max_samp = (bounds[1])[plot_num]

                hist_width = 0.01*(max_samp - min_samp)
                if hist_width == 0:
                    hist_width = 0.01*(more_sampled[:, dim_num].max())
                lower = min_samp - 2*hist_width
                upper = max_samp + 2*hist_width
                fig.add_trace(
                    go.Histogram(
                        name='Model samples',
                        x=more_sampled[:, dim_num],
                        histnorm='probability density',
                        showlegend=False,
                        xbins_size=hist_width,
                        marker_color=dist_colour,
                    ),
                    row=row,
                    col=col
                )

                if pop_params is not None:
                    fig.add_trace(
                        go.Scatter(
                            name='Typical',
                            x=[typs[dim_num]]*2,
                            y=[0, min(typs_height[dim_num], 1/hist_width)],
                            mode='lines',
                            line=dict(color='black', dash='dash'),
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )

                if param in individual_parameters.keys():
                    for i, x in enumerate(individual_parameters[param]):
                        if isinstance(model, chi.PooledModel):
                            height = 1/hist_width
                        elif isinstance(model, chi.LogNormalModel):
                            log_x = np.log(x)
                            height = min(np.exp(
                                const[dim_num] - log_x - (
                                    (log_x-log_typs[dim_num])**2
                                    / (2*omegas[dim_num]**2)
                                )
                            ), 100)
                        else:
                            adjusted_param = typs.copy()
                            adjusted_param[dim_num] = x
                            height = model.cdf(adjusted_param)

                        trace_col = np.reshape(ind_colour_selection, -1)[i]
                        fig.add_trace(
                            go.Scatter(
                                name='Individual Parameter',
                                x=[x]*2,
                                y=[0, height],
                                mode='lines',
                                line=go.scatter.Line(
                                    color=trace_col,
                                    width=0.5
                                ),
                                showlegend=False
                            ),
                            row=row,
                            col=col
                        )

                fig.update_xaxes(title_text=param, row=row, col=col)

                if row == 1:
                    fig.update_yaxes(
                        title_text='Probability',
                        row=row,
                        col=col
                    )

                fig.update_xaxes(range=[lower, upper], row=row, col=col)
                plot_num += 1

        fig.update_layout(
            template='plotly_white',
            width=1000,
            height=750,
        )
        return fig

    def plot_over_time(
            self, params, ind_params=None, show_data=False, doses=None,
            title=None, highlight_first=False, PK_PD=0
    ):
        # number of data points for simulations
        n_sim_ids = 1000
        n_times = 1000

        # determine number of parameters
        n_params = self.mech_model.n_parameters()

        # determine number of subplots and parameters for each subplot
        if isinstance(params, dict):
            subplots = len(params)
        else:
            subplots = 1
            params = {"": params}

        if ind_params is not None:
            if not isinstance(ind_params, dict):
                ind_params = {"": ind_params}
            if not (ind_params.keys() <= params.keys()):
                raise KeyError(
                    "The keys of ind params must match the keys of params"
                )
        # Initiailise graph
        if highlight_first and subplots > 1:
            n_rows = max(2, int((subplots-1)/3)+2)
            n_cols = 3
            specs = [
                [{"rowspan": 2, "colspan": 2}, None, {}],
                [None, None, {}]
            ] + [[{}, {}, {}]] * (n_rows-2)
        else:
            n_rows = int((subplots-1)/3)+1
            n_cols = min(subplots, 3)
            specs = [[{}]*n_cols] * n_rows
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=specs,
            subplot_titles=list(params.keys())
        )
        trace_count = 0
        for i_plot, subplot_name in enumerate(params):
            first_graph = i_plot == 0
            if highlight_first:
                if first_graph:
                    row = 1
                    col = 1
                elif i_plot == 1:
                    row = 1
                    col = 3
                else:
                    row = int(i_plot/3)+2
                    col = i_plot % 3 + 1
            else:
                row = int(i_plot/3)+1
                col = i_plot % 3 + 1
            plot_params = params[subplot_name]
            if ind_params is not None and subplot_name in ind_params.keys():
                plot_ind_params = ind_params[subplot_name]
            else:
                plot_ind_params = None
            if self.pop_model is not None:
                pop_params = np.empty((0, ))
                model_params = np.empty((0, ))
                if isinstance(self.pop_model, chi.ComposedPopulationModel):
                    pop_models = self.pop_model.get_population_models()
                else:
                    pop_models = [self.pop_model]
                il = 0
                for model in pop_models:
                    n_dims = model.n_dim()
                    if isinstance(model, chi.LogNormalModel):
                        typ = plot_params[il:il+n_dims]
                        omega = plot_params[il+n_dims:il+2*n_dims]
                        pop_params = np.concatenate(
                            (pop_params, np.log(typ), omega)
                        )
                        model_params = np.concatenate((model_params, typ))
                        il += 2*n_dims
                    else:
                        typ = plot_params[il:il+n_dims]
                        pop_params = np.concatenate((pop_params, typ))
                        model_params = np.concatenate((model_params, typ))
                        il += n_dims

                sim_parameters = self.pop_model.sample(
                    parameters=pop_params,
                    n_samples=n_sim_ids
                )
            else:
                sim_parameters = np.asarray([plot_params])
            if doses is None and self.data_set:
                dose_groups = self.dose_groups
                dose_amts = self.dose_amts
                n_per_dose = self.n_ids_per_dose
                dose_times = np.asarray(self.dose_times)
            elif doses is None:
                dose_groups = [1.0]
                if plot_ind_params is None:
                    n_per_dose = {1.0: 1}
                else:
                    n_per_dose = {1.0: plot_ind_params.shape[0]}
                dose_amts = [1.0]*n_per_dose[1.0]
                dose_times = [48.0*PK_PD]*n_per_dose[1.0]
            else:
                dose_groups = doses
                if plot_ind_params is None:
                    n_per_dose = dict(zip(doses, [1]*len(doses)))
                    dose_amts = doses
                else:
                    n_per_dose = dict(
                        zip(doses, [plot_ind_params.shape[0]]*len(doses))
                    )
                    dose_amts = np.flatten([doses]*plot_ind_params.shape[0])
                dose_times = [48.0*PK_PD]*len(dose_amts)

            d_iu = 0
            for i, group in enumerate(dose_groups):
                # Select the dose amts for patients in this dose_group
                d_il = d_iu
                d_iu = d_il + n_per_dose[group]

                # Calculate the average dose and time
                avg_dose = np.mean(dose_amts.iloc[d_il:d_iu])
                avg_dose_time = np.mean(dose_times[d_il:d_iu])
                self.mech_model.set_dosing_regimen(
                    dose=avg_dose, start=avg_dose_time, period=0
                )

                # times to plot curve
                if PK_PD == 0:
                    more_times = np.linspace(
                        start=avg_dose_time, stop=5+avg_dose_time, num=n_times
                    )
                else:
                    more_times = np.linspace(
                        start=0, stop=500+avg_dose_time, num=n_times
                    )
                # Simulate population distribution of measurements
                pop_samples = np.empty(shape=(n_sim_ids, n_times))

                fail_sim = False
                for i_sample, ind_sim_param in enumerate(sim_parameters):
                    try:
                        result = self.mech_model.simulate(
                            ind_sim_param[:n_params], more_times
                        )[PK_PD]
                        error_params = ind_sim_param[n_params:]
                        if self.error_model is None:
                            pop_samples[i_sample] = result
                        elif isinstance(self.error_model, chi.ErrorModel):
                            pop_samples[i_sample] = self.error_model.sample(
                                error_params[PK_PD:PK_PD+1], result
                            )[:, 0]
                        else:
                            error_model = self.error_model[PK_PD]
                            pop_samples[i_sample] = error_model.sample(
                                error_params[PK_PD:PK_PD+1], result
                            )[:,  0]
                    except myokit.SimulationError:
                        fail_sim = True
                if fail_sim:
                    Warning(
                        "Some simulations failed when determining percentiles"
                    )
                pop_samples = pop_samples[~np.isnan(pop_samples).any(axis=1)]

                trace_col = self.ind_colours[i, int(n_per_dose.at[group]/2)]
                if pop_samples.ndim == 2:
                    fifth = np.percentile(pop_samples, q=5, axis=0)
                    ninety_fifth = np.percentile(pop_samples, q=95, axis=0)

                    fig.add_trace(go.Scatter(
                        x=np.hstack(
                            [more_times, more_times[::-1]]
                        )-avg_dose_time,
                        y=np.hstack([fifth, ninety_fifth[::-1]]),
                        line=dict(width=0, color=trace_col),
                        fill='toself',
                        name='Population model',
                        legendgroup=group,
                        text=r"90% bulk probability",
                        hoverinfo='text',
                        showlegend=first_graph
                    ), row=row, col=col)

                # Plot the curve from the parameters provided
                more_values = self.mech_model.simulate(
                    model_params[:n_params], more_times
                )
                fig.add_trace(
                    go.Scatter(
                        x=more_times-avg_dose_time,
                        y=more_values[PK_PD],
                        mode='lines',
                        line=dict(color=trace_col, dash='dash'),
                        name='Typical simulation',
                        legendgroup=group,
                        legendgrouptitle={
                            'text': 'Dose '+str(group)+' '+self.dose_unit
                        },
                        showlegend=first_graph
                    ), row=row, col=col
                )

                # Plot individual traces
                if plot_ind_params is not None:
                    if self.data_set:
                        ids = self.df_PK_graph.loc[
                            self.df_PK_graph[self.dose_group_label] == group
                        ]['ID'].unique()
                        ind_params_dose = plot_ind_params[:len(ids)]
                        plot_ind_params = plot_ind_params[len(ids):]
                        ind_colours = self.ind_colours[i]
                    else:
                        ind_params_dose = plot_ind_params
                        ids = len(plot_ind_params)
                        ind_colours = [self.base_colour]*len(ids)

                    for i_pat, pat_param in enumerate(ind_params_dose):
                        ind_dose = dose_amts.iloc[d_il+i_pat]
                        ind_dose_time = dose_times[d_il+i_pat]
                        self.mech_model.set_dosing_regimen(
                            dose=ind_dose, start=ind_dose_time, period=0
                        )

                        # times to plot curve
                        if PK_PD == 0:
                            ind_times = np.linspace(
                                start=ind_dose_time,
                                stop=5+ind_dose_time,
                                num=n_times
                            )
                        else:
                            ind_times = np.linspace(
                                start=0, stop=500+ind_dose_time, num=n_times
                            )

                        ind_vals = self.mech_model.simulate(
                            pat_param[:n_params], ind_times
                        )
                        legend = (
                            (i_pat == int(n_per_dose.at[group]/2))
                            & first_graph
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=ind_times-ind_dose_time,
                                y=ind_vals[PK_PD],
                                name='Individual simulation',
                                legendgroup=group,
                                showlegend=legend,
                                mode="lines",
                                line=go.scatter.Line(
                                    color=ind_colours[i_pat], width=0.5
                                ),
                                opacity=0.5,
                            ), row=row, col=col
                        )

                # Plot the data
                if show_data:
                    if PK_PD == 0:
                        df_graph = self.df_PK_graph
                        label = self.PK_y_label
                    if PK_PD == 1:
                        df_graph = self.df_PD_graph
                        label = self.PD_y_label
                    ids = df_graph.loc[
                        df_graph[self.dose_group_label] == group
                    ]['ID'].unique()
                    for i_pat, ind in enumerate(ids):
                        data = df_graph.loc[df_graph['ID'] == ind]
                        dose_time = self.dose_times[ind]
                        legend = (
                            (i_pat == int(n_per_dose.at[group]/2))
                            & first_graph
                        )
                        fig.add_trace(go.Scatter(
                            x=data[self.time_label]-dose_time,
                            y=data[label],
                            mode='markers',
                            marker_color=self.ind_colours[i, i_pat],
                            name='Simulated data',
                            legendgroup=group,
                            showlegend=legend,
                            legendgrouptitle={
                                'text': 'Dose '+str(group)+' '+self.dose_unit
                            }
                        ), row=row, col=col)
            y_mins = []
            y_maxs = []
            for trace_data in fig.data[trace_count:]:
                data_above_zero = trace_data.y[trace_data.y > 0]
                y_mins.append(min(data_above_zero))
                y_maxs.append(max(data_above_zero))
                trace_count += 1
            y_min = min(y_mins)
            y_max = max(y_maxs)
            if PK_PD == 0:
                y_label = self.PK_y_label
                y_range = [
                    max(np.log10(y_min)-0.2, -4), min(np.log10(y_max)+0.2, 4)
                ]
                y_scale = "log"
                y_dtick = 1
                y_minor_tick = dict(dtick='D1', showgrid=True)
                x_dtick = 1
                x_minor_tick = None
            if PK_PD == 1:
                y_label = self.PD_y_label
                y_range = [
                    max(y_min*0.9, 0), min(y_max*1.1, 10e6)
                ]
                y_scale = "linear"
                y_dtick = 250
                y_minor_tick = None
                x_dtick = 72
                x_minor_tick = dict(dtick=24, showgrid=True)

            fig.update_yaxes(
                title_text=y_label,
                type=y_scale,
                minor=y_minor_tick,
                dtick=y_dtick,
                range=y_range,
                row=row, col=col
            )
            fig.update_xaxes(
                title_text=self.time_label,
                minor=x_minor_tick,
                dtick=x_dtick,
                row=row, col=col
            )
            first_graph = False

        width = min(n_cols, 3)*300+300
        height = n_rows*200+250
        fig.update_layout(
            title=title,
            template='plotly_white',
            width=width,
            height=height,
        )
        return fig

    def plot_param_sampling(
            self, posterior_samples, lines=None, param_names=None, legend=False
    ):
        az.style.use(["arviz-white",  "arviz-viridish"])
        vars = posterior_samples.data_vars
        fig, ax = plt.subplots(len(vars), 2, figsize=(8, 2*len(vars)))
        if param_names is None:
            labeller = azl.MapLabeller()
        else:
            labeller = azl.MapLabeller(
                var_name_map=dict(zip(vars, param_names))
            )

        def graph_tansform(params_to_trans):
            trans_param = params_to_trans
            if v_split[0] == "Log" and v_split[1] == "mean":
                trans_param[v] = np.exp(trans_param[v])
            return trans_param

        v_lines = None
        for i, v in enumerate(vars):
            v_split = v.split(' ')
            if len(v_split) == 1:
                colour_cycle = []
                for col in self.ind_colours.flatten():
                    colour_cycle.append(
                        pclrs.unconvert_from_RGB_255(pclrs.unlabel_rgb(col))
                    )
                plt.rcParams["axes.prop_cycle"] = cycler(color=colour_cycle)
            if lines is not None:
                v_lines = [(x, y, z) for x, y, z in lines if x == v]
            az.plot_trace(
                posterior_samples,
                var_names=[v],
                lines=v_lines,
                transform=graph_tansform,
                legend=(len(v_split) != 1) & legend,
                axes=ax[i:],
                labeller=labeller,
                show=False
            )
        return fig

    def optimise(
            self, function, start, fix=None, minimise=False, method='Powell',
            transform_ind=True
        ):

        # PINTS optimisers minimise.
        # To maximise the function, f, we will need to minimise -f
        if minimise:
            sign = 1
        else:
            sign = -1

        # Check whether there are any mixed effects parameters
        has_hierarchy = len(self.ME_param_args) > 0
        if has_hierarchy and transform_ind:
            # Find the arguments of the population level parameters and tile them for calculations on the individual level parameters
            typ_args = np.tile(self.ME_param_args[::2], self.n_ind)
            omega_args = np.tile(self.ME_param_args[1::2], self.n_ind)

            # Define function to convert from [individual params, population params] to [individual eta*omega, population params] and vice versa
            def transform_ind_to_eta(param, reverse=False):
                typ_params = param[typ_args]
                omega_params = param[omega_args]
                trans_param = param.copy()
                if reverse:
                    eta_params = param[:self.n_ind_params]
                    ind_params = np.exp((eta_params*omega_params + typ_params))
                    trans_param[:self.n_ind_params] = ind_params
                else:
                    ind_params = param[:self.n_ind_params]
                    eta_params = (np.log(ind_params) - typ_params)/omega_params
                    trans_param[:self.n_ind_params] = eta_params

                return param
        else:
            def transform_ind_to_eta(param, reverse=False):
                return param

        # Transform start point into [eta, pop]
        opt_start = start.copy()
        opt_start = transform_ind_to_eta(opt_start)
        if any(np.abs(transform_ind_to_eta(opt_start, reverse=True) - start) > 1e-10):
            raise ValueError(
                "Transform function is inaccurate: difference of" +
                str(transform_ind_to_eta(opt_start, reverse=True) - start)
            )

        if fix is not None:
            fix = np.asarray(fix)
            # If there are fixed params, determine what values need to be deleted and inserted
            if len(fix.shape) == 1:
                delete_arg = int(fix[0])
                insert_arg = int(fix[0])
                fix_values = fix[1]
            else:
                fix = fix[:, np.argsort(fix[0])]
                insert_arg = fix[0].astype(int)-np.arange(len(fix[0]))
                delete_arg = fix[0].astype(int)
                fix_values = fix[1]

            # Delete fixed values from the starting point
            opt_start = np.delete(opt_start, delete_arg)

            # Create the function to minimise
            def minimise_func(reduced_params):
                # insert the fixed values to the correct spots
                full_params_trans = np.insert(reduced_params, insert_arg, fix_values)
                # transform parameters back from [eta, pop] to [ind, pop]
                full_params = transform_ind_to_eta(full_params_trans, reverse=True)
                return sign*function(full_params)
        else:
            def minimise_func(full_params):
                full_params = transform_ind_to_eta(full_params, reverse=True)
                return sign*function(full_params)
        significant_change = 1e-11  # 1e-4
        max_iters = 10000
        max_unchanged_iters = 200
        if method == "PINTS" or method == "CMAES":
            if method == "PINTS":
                optimiser = pints.NelderMead(opt_start)
            elif method == "CMAES":
                optimiser = pints.CMAES(opt_start)
            fbest = float('inf')
            running = True
            iteration = 0
            unchanged_iters = 0
            while running:
                # Ask for points to evaluate
                xs = optimiser.ask()

                # Evaluate the function at these points
                fs = [minimise_func(x) for x in xs]

                # Tell the optimiser the evaluations; allowing it to update its
                # internal state.
                optimiser.tell(fs)

                # Check if new best found
                fnew = optimiser.fbest()
                if fnew < fbest:
                    # Check if this counts as a significant change
                    if np.abs(fnew - fbest) < significant_change:
                        unchanged_iters += 1
                    else:
                        unchanged_iters = 0

                    # Update best
                    fbest = fnew
                else:
                    unchanged_iters += 1
                iteration += 1
                # Check stopping criteria
                if iteration >= max_iters:
                    running = False
                if unchanged_iters >= max_unchanged_iters:
                    running = False

                # Check for optimiser issues
                if optimiser.stop():
                    running = False
            xbest = optimiser.xbest()
        else:
            options = {'maxiter': max_iters}
            # if method in ["Nelder-Mead"]:
            #     options['fatol'] = significant_change
            # if method in ["Powell", "L-BFGS-B", "TNC"]:
            #     options['ftol'] = significant_change # /np.abs(minimise_func(opt_start))
            result = minimize(minimise_func, opt_start, method=method, options=options)
            xbest = result.x
            fbest = result.fun

        if fix is not None:
            xbest = np.insert(xbest, insert_arg, fix_values)

        xbest = transform_ind_to_eta(xbest, reverse=True)
        # if not result.success:
        #     print(result.status, result.message)
        return xbest, sign*fbest

    def f_over_param_range_sequential(
            self, f, i_param, param_range, params_ref, pairwise=False,
            normalise=True, individual_parameters=False, n_evals=50,
            extrapolate=0, return_proj=False
    ):
        if pairwise:
            pass
        else:
            x_values = np.linspace(param_range[0], param_range[1], n_evals)

            # Start from the centre, useful for profile likelihoods
            lower_incl = x_values <= params_ref[i_param+self.n_ind_params]
            x_values_lower = x_values[lower_incl][::-1]
            x_values_upper = x_values[np.logical_not(lower_incl)]
            x_values = np.concatenate((x_values_lower, x_values_upper))
            ll_result = np.empty_like(x_values)
            param_values = np.empty((len(x_values), len(params_ref)))
            if return_proj:
                proj = np.empty((len(x_values), len(params_ref)))

            upper = False
            for x_val_set in (x_values_lower, x_values_upper):
                for i_x, x in enumerate(x_val_set):
                    if i_x<=1:
                        # Set current approximation of optimum parameters to MLE
                        curr = params_ref.copy()
                    elif extrapolate>0:
                        if i_x>extrapolate+1:
                            k = extrapolate
                        else:
                            k = (i_x-1)

                        if upper:
                            x_interp = x_val_set[i_x-1-k:i_x]
                            param_interp = param_values[i_x+len(x_values_lower)-1-k:i_x+len(x_values_lower)]
                        else:
                            x_interp = x_val_set[i_x-1-k: i_x]
                            param_interp = param_values[i_x-1-k: i_x]
                            x_interp = x_interp[::-1]
                            param_interp = param_interp[::-1]

                        extrapolator = make_interp_spline(
                            x_interp,
                            param_interp,
                            k=k, axis=0
                        )
                        curr = extrapolator(x)
                        # if i_x == len(x_values_lower):
                        #     # Reset start point for second half
                        #     extrapolator = make_interp_spline(
                        #         x_values[0:extrapolate+1], param_values[0:extrapolate+1],
                        #         k=extrapolate, axis=0
                        #     )

                    else:
                        curr = param_values[i_x+upper*len(x_values_lower)-1]
                    if return_proj:
                        proj[i_x+upper*len(x_values_lower)] = curr
                    param_values[i_x+upper*len(x_values_lower)], ll_result[i_x+upper*len(x_values_lower)] = f(x, i_param+self.n_ind_params, curr=curr)
                upper = True

            if normalise:
                l_star = np.max(ll_result)-1.92
                ll_result = ll_result - l_star  # Normalise the result

            # Sort the results for plotting
            sort = np.argsort(x_values)
            ll_result = ll_result[sort]
            x_values = x_values[sort]
            param_values = param_values[sort]


            if return_proj:
                proj = proj[sort]
                return x_values, ll_result, param_values, proj

            return x_values, ll_result, param_values

    def f_over_param_range_piecewise(
            self, f, i_param, param_range, params_anchors, pairwise=False,
            normalise=True, individual_parameters=False, n_evals=50,
            interpolate=1, return_proj=False
    ):
        if pairwise:
            pass
        else:
            x_values = np.asarray(params_anchors[0])
            ll_result =  np.asarray(params_anchors[1])
            param_values = np.asarray(params_anchors[2])
            if return_proj:
                proj_rec = param_values.copy()
            stop_criteria = [False]*len(x_values)
            max_opts = 100
            min_opts = 10
            n_del = 2

            while len(x_values) <= max_opts:
                x_insert = []
                insert_args = []
                ll_insert = []
                param_insert = []
                extrapolate = False
                # Determine where to add new points
                if x_values[0] !=  param_range[0]:
                    insert_args += [0, 0]
                    x_insert.append(param_range[0])
                    x_insert.append((param_range[0]+x_values[0])*0.5)
                    extrapolate = True
                for i in range(1, len(x_values)):
                    skip = stop_criteria[i-1] and stop_criteria[i]
                    if len(x_values) > min_opts+1 and skip:
                        continue
                    else:
                        insert_args += [i]
                        x_insert.append((x_values[i-1] + x_values[i])*0.5)
                if x_values[-1] !=  param_range[1]:
                    insert_args += [len(x_values), len(x_values)]
                    x_insert.append((param_range[1]+x_values[-1])*0.5)
                    x_insert.append(param_range[1])
                    extrapolate = True
                x_insert = np.asarray(x_insert)

                # Create intitial prediction using interpolation
                if extrapolate:
                    interpolator = interp1d(
                            x_values, param_values,
                            kind=interpolate, axis=0, fill_value='extrapolate'
                        )
                else:
                    interpolator = make_interp_spline(
                            x_values, param_values,
                            k=interpolate, axis=0
                        )
                    
                proj_values = interpolator(x_insert)
                if return_proj:
                    proj_rec = np.insert(proj_rec, insert_args, proj_values)

                # Maximise other params and Generate profile ll score
                for i, x in enumerate(x_insert):
                    param_value, ll = f(x, i_param+self.n_ind_params, curr=proj_values[i])
                    ll_insert.append(ll)
                    param_insert.append(param_value)

                arg_delete = None
                if len(x_values) >= min_opts:
                    # Check if the approximation of previous values is better

                    interpolator = make_interp_spline(
                            x_insert, param_insert,
                            k=interpolate, axis=0
                        )
                    # n_del += 1
                    error = np.sum(np.abs(interpolator(x_values) - param_values), axis=1)
                    error_new = np.sum(np.abs(proj_values - np.asarray(param_insert)), axis=1)
                    
                    stop_criteria = error <= 1e-2
                    arg_delete = np.argpartition(error, len(error)-n_del)[-n_del:] # np.argmax(error)
                    arg_delete = arg_delete[np.logical_not(stop_criteria[arg_delete])]
                    arg_delete += np.count_nonzero(np.tile(insert_args, (n_del, 1)).transpose() <= arg_delete, axis=0)

                    # max_error = 0
                    # for i, x in enumerate(x_values):
                    #     if x not in x_insert and not all(stop_criteria[i-1:i+1]):
                    #         stop_criteria[i] = False
                    #         mask = [True]*len(x_values)
                    #         view = slice(max(0, i-(interpolate+2)), min(i+(interpolate+3), len(x_values)+1))
                    #         mask[view] = [True]*len(mask[view])
                    #         mask[i] = False
                    #         interpolator = make_interp_spline(
                    #                 x_values[mask], param_values[mask],
                    #                 k=interpolate, axis=0
                    #             )
                    #         error = np.sum(np.abs(interpolator(x) - param_values[i]))
                    #         if  error <= 1e-2:
                    #             stop_criteria[i] = True
                    #         elif error > max_error:
                    #             max_error = error
                    #             arg_delete = i
                    # Check similarity between projection and optisised
                    # stop_insert = np.sum(np.abs(proj_values - np.asarray(param_insert)), axis=1) < 1e-2
                    stop_criteria = np.insert(stop_criteria, insert_args, error_new<= 1e-2) 
                else:
                    stop_criteria += [False]*len(x_insert)

                # Insert the new values into arrays
                x_values = np.insert(x_values, insert_args, x_insert)
                ll_result = np.insert(ll_result, insert_args, ll_insert)
                param_values = np.insert(param_values, insert_args, param_insert, axis=0)


                if arg_delete is not None:
                    # Slowly remove potential errors to smooth curve
                    x_values = np.delete(x_values, arg_delete)
                    ll_result = np.delete(ll_result, arg_delete)
                    param_values = np.delete(param_values, arg_delete, axis=0)
                    stop_criteria = np.delete(stop_criteria, arg_delete)

                if np.all(stop_criteria) and len(x_values) > min_opts+1:
                    break

            if normalise:
                max_score = np.max(ll_result)
                ll_result = ll_result - max_score  # Normalise the result

            if return_proj:
                return x_values, ll_result, param_values, proj_rec

            return x_values, ll_result, param_values

    def f_over_param_range_poly_approx(
            self, f, i_param, param_range, params_anchors, l_star, pairwise=False,
            individual_parameters=False, max_evals=50, interpolate=2, normalise=True
    ):
        if pairwise:
            pass
        else:
            x_values = np.asarray(params_anchors[0])
            ll_values = np.asarray(params_anchors[1])
            param_values = np.asarray(params_anchors[2])
            x_rec = x_values.copy()
            ll_rec = ll_values.copy()
            param_rec = param_values.copy()


            stop_criteria = [None]*2
            T_ident = 1e-3
            T_unident = 1e-4
            unchanged_iters_L = 0
            unchanged_iters_U = 0

            while len(x_rec) <= max_evals:
                # Solve LL(x)-L_star=0 approximating space as polynomial
                poly_spline = PPoly.from_spline(make_interp_spline(
                    x_values, ll_values - l_star,
                    k=interpolate, axis=0
                ))
                roots = np.sort(poly_spline.roots())

                # Find optimum of parameters at the roots.
                interpolator = make_interp_spline(
                        x_values, param_values,
                        k=interpolate, axis=0
                    )
                if stop_criteria[0] is None:
                    param_values[0], ll_values[0] = f(roots[0], i_param+self.n_ind_params, curr=interpolator(roots[0]))
                    x_values[0] = roots[0]

                    param_rec = np.insert(param_rec, 0, param_values[0], axis=0)
                    ll_rec = np.insert(ll_rec, 0, ll_values[0])
                    x_rec = np.insert(x_rec, 0, x_values[0])

                    if np.abs(ll_values[0]-l_star) < T_ident:
                        stop_criteria[0] = 'Ident'
                    elif np.abs(ll_rec[0] - ll_rec[1]) < T_unident:
                        unchanged_iters_L += 1
                        if unchanged_iters_L >= 5:
                            stop_criteria[0] = 'Unident'
                    else:
                        unchanged_iters_L = 0

                if stop_criteria[1] is None:
                    param_values[2], ll_values[2] = f(roots[1], i_param+self.n_ind_params, curr=interpolator(roots[1]))
                    x_values[2] = roots[1]
                    
                    param_rec = np.insert(param_rec, len(x_rec), param_values[2], axis=0)
                    ll_rec = np.insert(ll_rec, len(x_rec), ll_values[2])
                    x_rec = np.insert(x_rec, len(x_rec), roots[1])

                    if np.abs(ll_values[2]-l_star) < T_ident:
                        stop_criteria[1] = 'Ident'
                    elif np.abs(ll_rec[-1] - ll_rec[-2]) < T_unident:
                        unchanged_iters_U += 1
                        if unchanged_iters_U >= 5:
                            stop_criteria[1] = 'Unident'
                    else:
                        unchanged_iters_U = 0

                if all(stop_criteria):
                    break

            if normalise:
                max_score = np.max(ll_rec)
                ll_rec = ll_rec - l_star  # Normalise the result

            print("Termination status:", stop_criteria)

            # # Sort the results for plotting
            # sort = np.argsort(x_rec)
            # x_rec = x_rec[sort]
            # ll_rec = ll_rec[sort]
            # param_rec = param_rec[sort]

            return x_rec, ll_rec, param_rec

    def create_plot_func(self, function, params_ref, profile=False, profile_opts=None):
        def slice_function(param_value, param_arg, curr=None):
            slice_param = params_ref.copy()
            slice_param[param_arg] = param_value
            score = function(slice_param)
            return slice_param, score

        if profile:
            opts = {
                'method': 'Powell',
                'minimise': False,
                'transform_ind': True
            }
            if profile_opts is not None:
                opts.update(profile_opts)
            def profile_function(param_value, param_arg, curr):
                opt_param, score = self.optimise(
                    function,
                    curr,
                    fix=(param_arg, param_value),
                    minimise=opts['minimise'],
                    method=opts['method'],
                    transform_ind=opts['transform_ind']
                )
                return opt_param, score

            profile = True
            f = profile_function
        else:
            profile = False
            f = slice_function
        return f

    def plot_param_function(
            self, function, params_ref, profile=False, pairwise=False,
            individual_parameters=False, param_names=None, bounds=(None, None),
            force_bounds=(False, False), n_evals=50, profile_opts = None
    ):
        """
        Plot a function around a point in parameter space. Each parameter in
        the parameter space is varied one after another (or 2 at a time if
        pairwise is True).

        If profile is set to None, these are evaluated keeping all but the
        varied parameters at the params_ref. If profile is set to "maximum"
        (or "minimum"), the function is evaluated while all but the varied
        parameters are maximised (or minimised) using Nelder-Mead method.

        Parameters
        ----------
        function
            The function to plot. Must take a vector from the parameter space
            as a parameter and return a scalar.
        params_ref
            The parameter vector that the function is plotted around.
        profile
            Must be one of None, "maximum" or "minimum". Determines whether
            the function is optimised for the parameters that are not graphed.
        pairwise
            If False, only one dimensional plots are provided, If True, a grid
            of two-dimensional heat maps are provided.
        individual_parameters
            If False, individual parameters are ignored in graphs (but still
            must be included in params_ref). If True, individual parameters
            are included in graphs.
        param_names
            Parameter names to use for subplot titles. If not provided, names
            will be determined from the population model.
        bounds
            The range for each parameter to plot provided as a tuple of
            (upper_bounds, lower_bounds). This may be reduced for
            better view of the plot.
        n_evals
            The number of parameter values for each parameter to use for
            evaluations.

        Returns
        ----------
        Plotly figure with either 1 dimensional graphs of the function or a
        grid of pairwise heatmaps, with 1-d graphs on the diagonal.
        """

        # Create function
        slice_function = self.create_plot_func(
            function, params_ref, profile=None
        )

        if profile:
            f = self.create_plot_func(
                function, params_ref, profile=profile, profile_opts=profile_opts
            )
        else:
            f = slice_function
        ref_score = function(params_ref)

        # Set the parameter names
        n_params = int(len(params_ref)-self.n_ind_params)
        if (param_names is None) and (self.pop_model is not None):
            param_names = self.pop_model.get_parameter_names()
        elif param_names is None:
            param_names = np.char.add(
                ['Parameter ']*n_params,
                np.arange(1, n_params+1).astype(str)
            )
        # Create subplots
        if pairwise:
            n_rows = n_params
            n_cols = n_params
        else:
            n_rows = int((n_params+2)/3)
            n_cols = min(n_params, 3)
        fig = make_subplots(rows=n_rows, cols=n_cols)
        plot_num = 0

        # Set up colours
        pop_colour = 'darkgrey'
        if not individual_parameters:
            pop_colour = self.base_colour

        if self.data_set:
            ind_colour_selection = self.ind_colours
        else:
            ind_colour_selection = [self.base_colour]*self.n_ind

        # Set up bounds
        if bounds[0] is None:
            lower_bound = 0.5*params_ref[self.n_ind_params:]
        else:
            lower_bound = bounds[0]
        if bounds[1] is None:
            upper_bound = 1.5*params_ref[self.n_ind_params:]
        else:
            upper_bound = bounds[1]

        if isinstance(force_bounds[0], bool):
            force_lower_bounds = [force_bounds[0]]*len(lower_bound)
        else:
            force_lower_bounds = force_bounds[0]
        if isinstance(force_bounds[1], bool):
            force_upper_bounds = [force_bounds[1]]*len(lower_bound)
        else:
            force_upper_bounds = force_bounds[1]

        # Begin plotting

        # Plot 1-D graphs
        param_ranges = {}
        for i_param in range(0, n_params):
            print("Plot", plot_num+1, "/", n_params,
                  "1D graphs. Plotting", param_names[i_param])
            if pairwise:
                row = i_param+1
                col = i_param+1
            else:
                row = int(i_param/3)+1
                col = i_param % 3 + 1
            param_ref_i = params_ref[i_param+self.n_ind_params]

            # Refine x_values to a better view around the reference parameters
            x_min = lower_bound[i_param]
            i_check = 0
            j_check = 0
            view_aim = 3*1.92

            if not force_lower_bounds[i_param]:
                while i_check == 0 and j_check <= 20:
                    x_check = np.linspace(param_ref_i, x_min, 20)[1:]
                    for x in x_check:
                        _, score = slice_function(x, i_param+self.n_ind_params)
                        if score < (ref_score - view_aim):
                            x_min = x_check[i_check]
                            break
                        i_check += 1
                    j_check += 1
            x_max = upper_bound[i_param]
            i_check = 0
            j_check = 0
            if not force_upper_bounds[i_param]:
                while i_check == 0 and j_check <= 20:
                    x_check = np.linspace(param_ref_i, x_max, 20)[1:]
                    for x in x_check:
                        _, score = slice_function(x, i_param+self.n_ind_params)
                        if score < (ref_score - view_aim):
                            x_max = x_check[i_check]
                            break
                        i_check += 1
                    j_check += 1
            param_ranges[i_param] = (x_min, x_max)

            # Calculate function
            if profile_opts is None:
                extrapolate = 0
            elif 'projection' in profile_opts:
                extrapolate = profile_opts['projection']
            else:
                extrapolate = 0

            x_values, result = self.f_over_param_range_sequential(
                f, i_param, param_ranges[i_param], params_ref, n_evals=n_evals,
                extrapolate=extrapolate
            )[:2]
            args = [i_param+self.n_ind_params, i_param+self.n_ind_params]

            # Plot function
            fig.add_trace(
                go.Scatter(
                    name='Function',
                    x=x_values,
                    y=result,
                    showlegend=plot_num == 0,
                    mode='lines',
                    line=dict(color=pop_colour),
                ),
                row=row,
                col=col
            )

            # # Plot reference line
            # fig.add_trace(
            #     go.Scatter(
            #         name='Parameter Reference',
            #         x=[param_ref_i]*2,
            #         y=[np.min(result), param_ref_y],
            #         mode='lines',
            #         line=dict(color=pop_colour),
            #         showlegend=plot_num == 0
            #     ),
            #     row=row,
            #     col=col
            # )

            # Format axis
            if plot_num == 0 or not pairwise:
                fig.update_yaxes(
                    title_text='Normalised Function Value',
                    row=row,
                    col=col
                )
            else:
                fig.update_yaxes(
                    title_text="",
                    visible=False,
                    row=row,
                    col=col
                )
            if row == n_rows or not pairwise:
                fig.update_xaxes(
                    title_text=param_names[i_param],
                    row=row,
                    col=col
                )
            else:
                fig.update_xaxes(
                    title_text="",
                    visible=False,
                    row=row,
                    col=col
                )
            plot_num += 1

        # Plot 2D heatmaps
        if pairwise:
            print("Plotting 2D graphs.")
            for row in range(1, n_rows+1):
                for col in range(1, row):
                    ij_param = [col-1, row-1]
                    print("\t parameters:", param_names[ij_param[0]],
                          "and", param_names[ij_param[1]])
                    x_range = param_ranges[ij_param[0]]
                    y_range = param_ranges[ij_param[1]]

                    x_values = np.linspace(x_range[0], x_range[1], n_evals)
                    y_values = np.linspace(y_range[0], y_range[1], n_evals)
                    result = np.empty((n_evals, n_evals))

                    # Split up problem (useful for profile likelihood)
                    n_x_left = np.count_nonzero(
                        x_values <= params_ref[ij_param[0]+self.n_ind_params]
                    )
                    n_y_lower = np.count_nonzero(
                        y_values <= params_ref[ij_param[1]+self.n_ind_params]
                    )
                    left = range(n_x_left)[::-1]
                    right = range(n_x_left, len(x_values))
                    lower = range(n_y_lower)[::-1]
                    upper = range(n_y_lower, len(y_values))
                    ij_ranges = [
                        [left, lower],
                        [right, lower],
                        [left, upper],
                        [right, upper]
                    ]
                    # Fill each qurter of the results matrix
                    args = [
                        ij_param[0]+self.n_ind_params,
                        ij_param[1]+self.n_ind_params
                    ]
                    for i_range, j_range in ij_ranges:
                        if profile:
                            line_optimum = params_ref.copy()
                        for i_x in i_range:
                            x = x_values[i_x]
                            line_start = True
                            for j_y in j_range:
                                y = y_values[j_y]
                                if line_start and (profile is not None):
                                    if profile == "minimise":
                                        minimise = True
                                    else:
                                        minimise = False
                                    curr, result[j_y, i_x] = self.optimise(
                                        function,
                                        line_optimum,
                                        fix=[args, [x, y]],
                                        minimise=minimise
                                    )
                                    line_optimum = curr
                                    line_start = False
                                else:
                                    curr, result[j_y, i_x] = f([x, y], args)
                    max_score = np.max(result)
                    result = result - max_score  # Normalise result
                    # Plot Heatmap
                    fig.add_trace(
                        go.Heatmap(
                            name='Function',
                            z=result,
                            x=x_values,
                            y=y_values,
                            showlegend=plot_num == 1,
                            colorscale=self.heat_colour_scale,
                            zmin=-1.92,
                            zmax=+1.92
                        ),
                        row=row,
                        col=col
                    )
                    # Plot reference values
                    fig.add_trace(
                        go.Scatter(
                            name='Parameter Reference',
                            x=[params_ref[args[0]]]*2,
                            y=y_range,
                            mode='lines',
                            line=dict(color='black'),
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )
                    fig.add_trace(
                        go.Scatter(
                            name='Parameter Reference',
                            y=[params_ref[args[1]]]*2,
                            x=x_range,
                            mode='lines',
                            line=dict(color='black'),
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )

                    # Format axes
                    if col == 1:
                        fig.update_yaxes(
                            title_text=param_names[ij_param[1]],
                            row=row,
                            col=col
                        )
                    else:
                        fig.update_yaxes(
                            title_text="",
                            visible=False,
                            row=row,
                            col=col
                        )

                    if row == n_rows:
                        fig.update_xaxes(
                            title_text=param_names[ij_param[0]],
                            row=row,
                            col=col
                        )
                    else:
                        fig.update_xaxes(
                            title_text="",
                            visible=False,
                            row=row,
                            col=col
                        )

        fig.update_layout(
            template='plotly_white',
            width=1000,
            height=750,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig

    def plot_single_profile_ll(
            self, log_likelihood, i_param, params_ref,
            param_name=None, bounds=(None, None),
            force_bounds=(False, False), n_evals=50, profile_opts=None,
            show=["PLL", "ind pll", "ind param 0"]
    ):
        # Potential Show: "pll", "ind pll", "param i", "ind param i", "deriv i"

        # Set name if not provided
        if (param_name is None) and (self.pop_model is not None):
            param_name = self.pop_model.get_parameter_names()[i_param]
        elif param_name is None:
            param_name = 'Parameter ' + str(i_param+1)

        # Create subplots
        n_rows = len(show)
        n_cols = 1
        fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True)

        # Set up colours
        pop_colour = 'darkgrey'
        if self.data_set:
            ind_colour_selection = self.ind_colours
        else:
            ind_colour_selection = [self.base_colour]*self.n_ind
        
        result = log_likelihood.result[i_param+self.n_ind_params]
        if len(result)==0:
            log_likelihood.run(i_param+self.n_ind_params, opts=profile_opts)
            result = log_likelihood.result[i_param+self.n_ind_params]
        bounds = log_likelihood.param_range[i_param+self.n_ind_params][0]
        x_values = np.linspace(bounds[0], bounds[1], n_evals)
        pll_values = result['profile ll'](x_values)
        param_values = result['profile param'](x_values)
        y_ax_names = []
        for row, graph_type in enumerate(show):
            if graph_type.startswith("PLL"):
                fig.add_trace(
                    go.Scatter(
                        name='Profile Log-likelihood',
                        x=x_values,
                        y=pll_values,
                        mode='lines',
                        line=dict(color=self.base_colour),
                    ),
                    row=row+1,
                    col=1
                )
                if graph_type == "PLL points":
                    optimiser_points = result['opt points']
                    fig.add_trace(
                        go.Scatter(
                            name='Optimiser Evaluations',
                            x=optimiser_points[0],
                            y=optimiser_points[1],
                            mode='markers',
                            marker=dict(
                                color=self.base_colour,
                                symbol="star-diamond",
                                opacity=0.5
                            ),
                        ),
                        row=row+1,
                        col=1
                    )
                elif graph_type == "PLL projection":
                    x_points = result['opt points'][0]
                    proj_params = result['projected optimum']
                    proj_LL = []
                    for param in proj_params:
                        proj_LL.append(log_likelihood(param))
                    fig.add_trace(
                        go.Scatter(
                            name='Projected PLL',
                            x=x_points[0],
                            y=proj_LL[1],
                            mode='markers',
                            marker=dict(
                                color=self.base_colour,
                                symbol="circle",
                                opacity=0.5
                            ),
                        ),
                        row=row+1,
                        col=1
                    )
                y_ax_names.append("Log-Likelihood")
            elif graph_type == "ind PLL":
                # create matrix of pointwise loglikelihoods of shape
                # (n_x_values, n_ind)
                ind_pll = np.empty((x_values.shape[0], self.n_ind))
                for i_vec, param_vec in enumerate(param_values):
                    point_pll = np.array(log_likelihood.compute_pointwise_ll(
                        param_vec, per_individual=True
                    ))
                    full_ll = log_likelihood(param_vec)
                    if np.abs(np.sum(point_pll) - full_ll) > 1e-3:
                        raise ValueError(
                            "Incorrect pointwise log-likelihood calculated " +
                            "for " + str(i_vec) + "th parameter vector:"
                            + str(np.sum(point_pll)) + "!=" + str(full_ll)
                        )
                    if point_pll.shape != (self.n_ind, ):
                        raise ValueError(
                            "Individual log-likelihood is wrong shape. Should" +
                            " be " + str((self.n_ind, )) + "but is " +
                            str(point_pll.shape)
                        )
                    ind_pll[i_vec] = point_pll
                ind_pll -= np.array(log_likelihood.compute_pointwise_ll(
                    params_ref, per_individual=True
                ))
                for i_ind in range(0, self.n_ind):
                    ind_colour = ind_colour_selection.flatten()[i_ind]
                    fig.add_trace(
                        go.Scatter(
                            name='Individual Log-likelihoods',
                            x=x_values,
                            y=ind_pll[:, i_ind],
                            mode='lines',
                            line=dict(color=ind_colour, width=1),
                            showlegend=False
                        ),
                        row=row+1,
                        col=1
                    )
            elif graph_type.startswith("ind param"):
                param_args = graph_type.split()[2:]
                for i_param in param_args:
                    i_param = int(i_param)
                    # ind_param = np.arange(self.n_ind_params)[
                    #     i_param::int(len(self.ME_param_args)/2)
                    # ]
                    for i_ind in range(0, self.n_ind):
                        ind_colour = ind_colour_selection.flatten()[i_ind]
                        ind_param_arg = (
                            i_param + i_ind*int(len(self.ME_param_args)/2)
                        )
                        fig.add_trace(
                            go.Scatter(
                                name='Individual Parameter optimised',
                                x=x_values,
                                y=(
                                    param_values[:, ind_param_arg] -
                                    params_ref[ind_param_arg]
                                ),
                                mode='lines',
                                line=dict(color=ind_colour, width=1),
                                showlegend=False
                            ),
                            row=row+1,
                            col=1
                        )

                    typ_param = self.ME_param_args[i_param*2]
                    norm_res = (
                        np.exp(param_values[:, typ_param])
                        - np.exp(params_ref[typ_param])
                    )
                    fig.add_trace(
                        go.Scatter(
                            name='Population Parameter optimised',
                            x=x_values,
                            y=norm_res,
                            mode='lines',
                            line=dict(color=pop_colour),
                            showlegend=False
                        ),
                        row=row+1,
                        col=1
                    )

            elif graph_type.startswith("deriv"):
                y_deriv = {0: pll_values}
                x_deriv = {0: x_values}
                n_deriv = int(graph_type[-1])
                for deriv in range(1, n_deriv+1):
                    x = x_deriv[deriv-1]
                    dy = np.diff(y_deriv[deriv-1], 1)
                    dx = np.diff(x, 1)
                    y_deriv[deriv] = dy/dx
                    x_deriv[deriv] = 0.5*(x[:-1]+x[1:])
                fig.add_trace(
                    go.Scatter(
                        name=str(deriv)+'th Derivative',
                        x=x_deriv[n_deriv],
                        y=y_deriv[n_deriv],
                        mode='lines',
                        line=dict(color=self.base_colour),
                        showlegend=False
                    ),
                    row=row+1,
                    col=1
                )

        fig.update_layout(
            template='plotly_white',
            width=500,
            height=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
