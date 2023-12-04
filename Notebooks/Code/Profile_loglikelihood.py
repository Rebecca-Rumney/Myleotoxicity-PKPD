import pandas
from Code.PD_model import PintsPDFribergLinE as Pints_PD_model
from Code.Likelihoods import ConstantAndMultiplicativeGaussianLogLikelihood
import pints
import nlopt
import numpy as np
# from scipy.optimize import minimize, brentq
import scipy.optimize as spopt
from scipy.interpolate import make_interp_spline, PPoly, Akima1DInterpolator
from scipy.interpolate import PchipInterpolator
import warnings


class ProfileLogLikelihood():

    def __init__(self, log_likelihood, params_ref, opts=None) -> None:
        self.set_options(opts, reset=True)
        self.ll = log_likelihood
        self.slice_func = self.create_func(
            self.ll, params_ref, profile=False
        )
        if self.opts['slice']:
            self.f = self.slice_func
        else:
            self.f = self.create_func(
                self.ll, params_ref, profile=True
            )
        self.MLE = params_ref.copy()
        self.ref_score = self.ll(params_ref)
        self.l_star = self.ref_score - self.opts['alpha']
        self.param_range = {}
        self.result = {}

        def all_subclasses(cls):
            return set(cls.__subclasses__()).union(
                [s for c in cls.__subclasses__() for s in all_subclasses(c)])

        opts_subclasses = all_subclasses(pints.Optimiser)
        self.pints_optimisers = dict(zip(
            [cls.__name__ for cls in opts_subclasses],
            opts_subclasses
        ))

    def run(self, i_param, n_points=100, opts=None):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)
        if i_param not in self.param_range.keys():
            self.set_param_range([i_param], adapt=True)

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
        self, func, start, fix=None, opts=None
    ):
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)

        minimise = local_opts['minimise']
        transformation = local_opts['transformation']
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
            # If there are fixed params, determine what values need to be
            # deleted and inserted
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

        if local_opts['lower_bounds'] is None:
            LB = None
        elif np.isscalar(local_opts['lower_bounds']):
            LB = local_opts['lower_bounds']*np.ones(len(opt_start))
        elif np.array(local_opts['lower_bounds']).ndim == 1:
            LB = local_opts['lower_bounds']
        
        if LB is not None:
            out_L_bounds = opt_start < LB
            if any(out_L_bounds):
                opt_start[out_L_bounds] = LB[out_L_bounds] + 1e-10

        if local_opts['upper_bounds'] is None:
            UB = None
        elif np.isscalar(local_opts['upper_bounds']):
            UB = local_opts['upper_bounds']*np.ones(len(opt_start))
        elif np.array(local_opts['upper_bounds']).ndim == 1:
            UB = local_opts['upper_bounds']
        if UB is not None:
            out_U_bounds = opt_start > UB
            if any(out_U_bounds):
                opt_start[out_U_bounds] = UB[out_U_bounds] - 1e-10

        if opt_package == "pints":
            if (UB is not None) and (LB is not None):
                bounds = pints.RectangularBoundaries(LB, UB)
            else:
                bounds = None
        if opt_package == "scipy":
            if LB is None:
                LB_sp = [None]*len(opt_start)
            else:
                LB_sp = LB
            if UB is None:
                UB_sp = [None]*len(opt_start)
            else:
                UB_sp = UB
            bounds = zip(LB_sp, UB_sp)

        if opt_package == "pints":
            if opt_method not in self.pints_optimisers.keys():
                raise NameError(
                    "pints does not have optimiser " + opt_method + ". " +
                    "Perhaps you need to change 'opt_package' option to " +
                    "'scipy'."
                )
            else:
                optimiser = self.pints_optimisers[opt_method]

            xbest, fbest = pints.fmin(
                minimise_func, opt_start, max_iter=max_iter,
                max_unchanged=max_unchanged, method=optimiser,
                boundaries=bounds
            )
        elif opt_package == "scipy":
            options = {}
            if max_iter is not None:
                options['maxiter'] = max_iter
            if opt_method == 'basinhopping':
                result = spopt.basinhopping(minimise_func, opt_start, minimizer_kwargs=dict(
                    method='Powell', options=options, seed=local_opts['random_seed']
                ))
            elif opt_method == 'differential_evolution':
                if LB is None:
                    LB = 1e-2*opt_start
                if UB is None:
                    UB = 1e2*opt_start
                result = spopt.differential_evolution(
                    minimise_func, bounds=np.transpose([LB, UB]), maxiter=max_iter,
                    popsize=30, mutation=(0.75, 1.25), recombination=0.5,
                    seed=local_opts['random_seed']
                )
            else:
                result = spopt.minimize(
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
        # options for double checking some values
        check_opts = opts.copy()
        # if local_opts['opt_package'] == 'scipy':
        #     check_opts['optimiser'] = 'differential_evolution'
        check_opts['opt_package'] = 'pints'
        check_opts['optimiser'] = 'CMAES'

        param_range = self.param_range[i_param][0]
        x_rec = np.linspace(
            param_range[0], param_range[1], local_opts['max_N_opts']
        )

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
        x_check = []
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
                    x, i_param, curr=curr, opts=opts
                )
                # If discrepencies are large, double check with global optimiser
                param_error = np.sum(
                    np.abs((param_rec[LU][i_x]-curr)/curr)
                )/len(curr)
                if param_error > 3 or not np.isfinite(ll_rec[LU][i_x]):
                    x_check.append(x)
                    param_rec[LU][i_x], ll_rec[LU][i_x] = self.f(
                        x, i_param, curr=self.MLE, opts=check_opts
                    )
            LU = 1

        x_rec = np.concatenate((x_lower[::-1], x_upper))
        ll_rec = np.concatenate((ll_rec[0][::-1], ll_rec[1]))
        param_rec = np.concatenate((param_rec[0][::-1], param_rec[1]))
        proj_rec = np.concatenate((proj_rec[0][::-1], proj_rec[1]))

        if len(x_check)>0:
            print("Used global optimisation on shape points " + str(x_check))
        if local_opts['normalise']:
            ll_rec = ll_rec - self.l_star  # Normalise the result
        finite_results = np.isfinite(ll_rec)
        param_interpolator = PchipInterpolator(
                x_rec[finite_results], param_rec[finite_results], axis=0
            )
        ll_interpolator = PchipInterpolator(
                x_rec[finite_results], ll_rec[finite_results]
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
        param_rec[0], ll_rec[0] = self.f(x_rec[0], i_param, curr=self.MLE, opts=opts)
        param_rec[2], ll_rec[2] = self.f(x_rec[2], i_param, curr=self.MLE, opts=opts)

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
                param_value, ll = self.f(x, i_param, curr=proj_values[i], opts=opts)
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
        # options for double checking some values
        check_opts = opts.copy()
        # if local_opts['opt_package'] == 'scipy':
        #     check_opts['optimiser'] = 'differential_evolution'
        # elif local_opts['opt_package'] == 'pints':
        #     check_opts['optimiser'] = 'CMAES'
        check_opts['opt_package'] = 'pints'
        check_opts['optimiser'] = 'CMAES'

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
            x_check = []
            for x in x_shape_points:
                ll_interpolator = PchipInterpolator(
                    x_rec[np.isfinite(ll_rec)], ll_rec[np.isfinite(ll_rec)],
                    extrapolate=True
                )
                param_interp = param_interpolator(x)
                param, ll = self.f(x, i_param, curr=param_interp, opts=opts)
                # If discrepencies are large, double check with global optimiser
                ll_error = np.abs(
                    ((ll-self.l_star)-ll_interpolator(x))/ll_interpolator(x)
                )
                if ll_error > 1.75:
                    x_check.append(x)
                    param, ll = self.f(
                        x, i_param, curr=self.MLE, opts=check_opts
                    )
                if local_opts['normalise']:
                    ll = ll - self.l_star
                insert_arg = np.searchsorted(x_rec, x)
                param_rec = np.insert(param_rec, [insert_arg], [param], axis=0)
                x_rec = np.insert(x_rec, insert_arg, x)
                ll_rec = np.insert(ll_rec, insert_arg, ll)
            if len(x_check)>0:
                print("Used global optimisation on shape points " + str(x_check))
            # result['opt points'] = (x_rec, ll_rec, param_rec)

        # weights = np.ones(len(x_rec))
        # weights[np.isin(x_rec, x_CI)] = 50
        # weights[x_rec == self.MLE[i_param]] = 200

        fin_result = np.isfinite(ll_rec)
        # if np.any(inf_result):
        #     ll_rec[inf_result] = 2
        #     weights[inf_result] = 0
        ll_interpolator = Akima1DInterpolator(
                x_rec[fin_result], ll_rec[fin_result],
                # k=2, w=weights, s=5e-2*len(weights)
            )
        x_unique, unique_args = np.unique(
            x_rec, return_index=True
        )
        # ll_interpolator = PchipInterpolator(x_rec, ll_rec, extrapolate=True)
        param_interpolator = PchipInterpolator(
                x_unique, param_rec[unique_args], axis=0, extrapolate=True
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

        # options for double checking values
        check_opts = opts.copy()
        # if local_opts['opt_package'] == 'scipy':
        #     check_opts['optimiser'] = 'differential_evolution'
        # elif local_opts['opt_package'] == 'pints':
        #     check_opts['optimiser'] = 'CMAES'
        check_opts['opt_package'] = 'pints'
        check_opts['optimiser'] = 'CMAES'

        # First step: Estimate CI using slice function
        param_range = self.param_range[i_param][0]
        x_CI = [param_range[0], self.MLE[i_param], param_range[1]]
        param_CI = [None, self.MLE, None]
        ll_CI = [None, self.ref_score, None]

        def solve_slice(x, L_U):
            param_CI[L_U+1], ll = self.slice_func(x, i_param)
            return ll - self.l_star
        x_CI[0] = spopt.brentq(solve_slice, x_CI[0], x_CI[1], args=(-1))
        x_CI[2] = spopt.brentq(solve_slice, x_CI[1], x_CI[2], args=(1))

        # Optimise at this point to find the log-likelihood
        param_CI[0], ll_CI[0] = self.f(x_CI[0], i_param, curr=param_CI[0], opts=opts)
        param_CI[2], ll_CI[2] = self.f(x_CI[2], i_param, curr=param_CI[2], opts=opts)

        x_rec = x_CI.copy()
        ll_rec = ll_CI.copy()
        param_rec = param_CI.copy()
        proj_rec = [self.MLE, self.MLE, self.MLE]

        identifiable_CI = [None]*2
        # counts for iterations where it is diverging or converging slowly
        divert_L = 0
        divert_U = 0
        slow_conv_L = 0
        slow_conv_U = 0
        x_check = []

        # Iteratively estimate CI approximating space as polynomial
        while len(x_rec) <= self.opts['max_N_opts']:
            ll_CI = np.array(ll_CI)
            # Solve LL(x)-L_star=0 assuming polynomial

            # Check ll is finite, if not replace with finite values
            if not all(np.isfinite(ll_CI)):
                param_interp = False
                ll_CI[np.isinf(ll_CI)] = (np.sign(ll_CI)*1e15)[np.isinf(ll_CI)]
                ll_CI[np.isnan(ll_CI)] = -1e15
            elif any(ll_CI > ll_CI[1]):
                param_interp = False
            else:
                param_interp = True

            poly_spline = PPoly.from_spline(make_interp_spline(
                x_CI, ll_CI - self.l_star,
                k=local_opts['interp'], axis=0
            ))
            roots = np.sort(poly_spline.roots())

            if len(roots) == 0:
                # This can only happen if one of the confidence intervals has
                # a greater likelihood than the MLE
                ll_fake = ll_CI.copy()
                ll_fake[ll_fake > ll_fake[1]] = ll_fake[1] - np.abs(ll_fake[1])*0.1
                poly_spline = PPoly.from_spline(make_interp_spline(
                    x_CI, ll_fake - self.l_star,
                    k=local_opts['interp'], axis=0
                ))
                roots = np.sort(poly_spline.roots())
            # Approximate maximum parameters.
            if param_interp:
                interpolator = make_interp_spline(
                        x_CI, param_CI,
                        k=local_opts['interp'], axis=0
                    )
            if identifiable_CI[0] is None:
                recalc_root = False
                if ll_CI[0] > self.ref_score:
                    warnings.warn(
                        "MLE is not accurate, found score above the maximum."
                    )
                    recalc_root = True
                if recalc_root or roots[0]<param_range[0]:
                    roots[0] = (x_CI[0]+param_range[0])*0.5
                if param_interp:
                    approx_param = interpolator(roots[0])
                else:
                    approx_param = self.MLE

                # Calculate true max params and likelihood at the lower root
                param_CI[0], ll_CI[0] = self.f(
                    roots[0], i_param, curr=approx_param, opts=opts
                )
                x_CI[0] = roots[0]

                # If discrepencies are large, double check with global optimiser
                param_error = np.sum(
                    np.abs((param_CI[0]-approx_param)/approx_param)
                )/len(approx_param)
                if param_error > 3:
                    x_check.append(roots[0])
                    param_CI[0], ll_CI[0] = self.f(
                        roots[0], i_param, curr=self.MLE, opts=check_opts
                    )
                
                # Record these values
                param_rec = np.insert(param_rec, 0, param_CI[0], axis=0)
                proj_rec = np.insert(proj_rec, 0, approx_param, axis=0)
                ll_rec = np.insert(ll_rec, 0, ll_CI[0])
                x_rec = np.insert(x_rec, 0, x_CI[0])

                # Check termination status
                if x_rec[0] < x_rec[1]:
                    diverging = ll_rec[1] - ll_rec[0] < local_opts['T_unident']
                else:
                    diverging = ll_rec[0] - ll_rec[1] < local_opts['T_unident']

                if np.abs(ll_CI[0]-self.l_star) < local_opts['T_ident']:
                    identifiable_CI[0] = 'Ident'
                elif diverging:
                    divert_L += 1
                    if divert_L >= 5:
                        identifiable_CI[0] = 'Unident'
                elif np.abs(ll_rec[1] - ll_rec[0]) < np.abs(ll_CI[0]-self.l_star):
                    slow_conv_L += 1
                    if slow_conv_L >= 5:
                        warnings.warn(
                            "Convergence is slow, quadratic approximation " +
                            "may not be appropriate"
                        )
                else:
                    divert_L = 0
                    slow_conv_L = 0

            if identifiable_CI[1] is None:
                if ll_CI[2] > self.ref_score:
                    warnings.warn(
                        "MLE is not accurate, found score above the maximum."
                    )
                    recalc_root = True
                if recalc_root or roots[1]>param_range[1]:
                    roots[1] = (x_CI[2]+param_range[1]*0.5)
                if param_interp:
                    approx_param = interpolator(roots[1])
                else:
                    approx_param = self.MLE

                # Calculate true max params and likelihood at the upper root
                param_CI[2], ll_CI[2] = self.f(
                    roots[1], i_param, curr=approx_param, opts=opts
                )
                x_CI[2] = roots[1]

                # If discrepencies are large, double check with global optimiser
                param_error = np.sum(
                    np.abs((param_CI[2]-approx_param)/approx_param)
                )/len(approx_param)
                if param_error > 3:
                    x_check.append(roots[1])
                    param_CI[2], ll_CI[2] = self.f(
                        roots[1], i_param, curr=self.MLE, opts=check_opts
                    )

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
                if x_rec[-1] > x_rec[-2]:
                    diverging = ll_rec[-2] - ll_rec[-1] < local_opts['T_unident']
                else:
                    diverging = ll_rec[-1] - ll_rec[-2] < local_opts['T_unident']

                if np.abs(ll_CI[2]-self.l_star) < local_opts['T_ident']:
                    identifiable_CI[1] = 'Ident'
                elif diverging:
                    divert_U += 1
                    if divert_U >= 5:
                        identifiable_CI[1] = 'Unident'
                elif np.abs(ll_rec[-2]-ll_rec[-1]) < np.abs(ll_CI[2]-self.l_star):
                    slow_conv_U += 1
                    if slow_conv_U >= 5:
                        warnings.warn(
                            "Convergence is slow, quadratic approximation " +
                            "may not be appropriate"
                        )
                else:
                    divert_U = 0
                    slow_conv_U = 0
            
            if all(identifiable_CI):
                break

        if len(x_check)>0:
            print("Used global optimisation for CI finding on points " + str(np.sort(x_check)))
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
                'max_unchanged': 200,
                'lower_bounds': None, # [0, 1e10],
                'upper_bounds': None,
                'random_seed': None
            }
        if opts is not None:
            self.opts.update(opts)

    def create_func(self, function, params_ref, profile=True):
        if profile:
            def profile_function(param_value, param_arg, curr, opts=None):
                opt_param, score = self.optimise(
                    function,
                    curr,
                    fix=[param_arg, param_value],
                    opts = opts
                )
                return opt_param, score
            f = profile_function
        else:
            def slice_function(param_value, param_arg, curr=None, opts=None):
                slice_param = params_ref.copy()
                slice_param[param_arg] = param_value
                score = function(slice_param)
                return slice_param, score
            f = slice_function
        return f

    def max_likelihood_estimate(self, n_runs=1, param_start=None, opts=None):
        """
        Finds and sets the maximum likelihood estimate (MLE).

        Parameters
        ----------

        n_runs
            The number of runs of the optimisation to determine the MLE. All
            results are retuned but only the best result is saved as the MLE.

        param_start
            The starting point of the optisation. NDArray of shape (n_params, )
            or (n_runs, n_params). If none is given, the previous MLE will be
            used
        """
        local_opts = self.opts.copy()
        if opts is not None:
            local_opts.update(opts)

        n_params = len(self.MLE)
        if param_start is None:
            param_start = np.tile(self.MLE, n_runs)
        elif len(np.array(param_start).shape) == 1:
            if len(param_start) != n_params:
                raise TypeError(
                    "Start parameters are not of correct size. Must be of " +
                    "shape " + str((n_params, )) + " or " +
                    str((n_runs, n_params)) + "."
                )
            param_start = np.tile(param_start, n_runs)
        elif np.array(param_start).shape != (n_runs, n_params):
            raise TypeError(
                "Start parameters are not of correct size. Must be of shape "
                + str((n_params, )) + " or " + str((n_runs, n_params)) + "."
            )

        MLE = np.empty_like(param_start)
        LL_scores = np.empty(n_runs)
        for i_param, param_v in enumerate(param_start):
            MLE[i_param], LL_scores[i_param] = self.optimise(
                self.ll, param_v, opts=opts
            )
        if local_opts['minimise']:
            best = np.argmin(LL_scores)
        else:
            best = np.argmax(LL_scores)

        self.MLE = MLE[best]
        self.ref_score = LL_scores[best]
        self.l_star = self.ref_score - self.opts['alpha']
        self.slice_func = self.create_func(
            self.ll, MLE[best], profile=False
        )

        return MLE, LL_scores

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


class profile_loglikelihood():
    """
        This class is depreceated, please use ProfileLogLikelihood
    """
    def __init__(self, log_likelihood, optimum_point, bounds):
        self.log_likelihood = log_likelihood
        self.optimum_point = optimum_point
        self.bounds = bounds
        self.log_outbounds = []
        self.base_opt = self.create_optimiser(len(optimum_point))
        self.timer = pints.Timer()
        self._n = 0
        self.frac_done = 0

    def create_optimiser(self, num_param):
        algorithm = nlopt.LN_BOBYQA
        opt = nlopt.opt(algorithm, num_param)
        opt.set_lower_bounds([0]*num_param)
        unchanged_threshold = 1e-4
        opt.set_ftol_abs(unchanged_threshold)
        opt.set_xtol_rel(1e-4)
        opt.set_max_objective(self.obj_func)
        return opt

    def reset_outbounds(self):
        log = self.log_outbounds
        self.log_outbounds = []
        return log

    def obj_func(self, x, grad):
        time_stamp = self.timer.time()
        print(
            "\r" + str(np.round(self.frac_done*100, 2)) +
            "% complete," + " time elapsed: " +
            self.timer.format(time_stamp),
            sep=' ',
            end='        ',
            flush=True
        )
        return self.log_likelihood(x)

    def run_optimisation(self, PL_bounds, point):
        # if point ended up out of bounds, nudge back into bounds
        rel_indices = np.logical_not(PL_bounds[0] == PL_bounds[1])
        outbounds = (
            np.logical_and(rel_indices, point <= PL_bounds[0, :]),
            np.logical_and(rel_indices, point >= PL_bounds[1, :])
        )
        check = any(np.logical_or(outbounds[0], outbounds[1]))
        if check:
            point[outbounds[0]] = 1.0001*(PL_bounds[0, :])[outbounds[0]]
            point[outbounds[1]] = 0.9999*(PL_bounds[1, :])[outbounds[1]]
            self.log_outbounds.append(point)

        # create optimiser
        opt = nlopt.opt(self.base_opt)
        opt.set_lower_bounds(PL_bounds[0, :])
        opt.set_upper_bounds(PL_bounds[1, :])

        # Optimise
        point_new = opt.optimize(point)
        loglike = opt.last_optimum_value()
        return point_new, loglike

    def single_prof_ll(self, param_index, param_range):
        # initialise point, bounds and profile likelihood values
        profile_likelihood = []
        point = self.optimum_point.copy()
        PL_bounds = self.bounds.copy()
        i = 0

        for theta_i in param_range:
            self.frac_done = self._n + (i/len(param_range))*0.5
            # ensure the parameter theta_i is fixed in the optimisation:
            PL_bounds[0, param_index] = theta_i
            PL_bounds[1, param_index] = theta_i
            # set fixed point
            point[param_index] = theta_i
            point, loglike = self.run_optimisation(PL_bounds, point)
            profile_likelihood.append([theta_i, loglike])
            i += 1
        return np.asarray(profile_likelihood)

    def pair_prof_ll(self, param_indexes, param_ranges):
        point_line = self.optimum_point.copy()
        PL_bounds = self.bounds.copy()

        index_1 = param_indexes[0]
        index_2 = param_indexes[1]
        range_1 = param_ranges[0]
        range_2 = param_ranges[1]

        profile_likelihood = np.zeros((len(range_1), len(range_2)))

        for i, theta_1 in enumerate(range_1):
            point = point_line.copy()
            # ensure the parameter theta_i is fixed in the optimisation
            PL_bounds[0, index_1], PL_bounds[1, index_1] = theta_1, theta_1

            for j, theta_2 in enumerate(range_2):
                self.frac_done = self._n + (
                    i/len(range_1) + j/(len(range_1)*len(range_2))
                )*0.25
                # ensure the parameter theta_j is fixed in the optimisation
                point[index_1], point[index_2] = theta_1, theta_2
                PL_bounds[0, index_2], PL_bounds[1, index_2] = theta_2, theta_2

                point, loglike = self.run_optimisation(PL_bounds, point)
                profile_likelihood[i, j] = loglike

                if j == 0:
                    point_line = point.copy()
        return profile_likelihood

    def save_single_profiles(
        self, save_file,
        tot_evals,
        indices=None,
        param_names=None,
        log_scale=False
    ):
        self.per_done = 0
        if param_names is None:
            param_names = np.char.add(
                np.asarray(["Parameter "]*len(self.optimum_point)),
                np.arange(0, len(self.optimum_point)).astype(str)
            )
        if indices is None:
            indices = range(0, len(self.optimum_point))
        if type(log_scale) is bool:
            log_scale = [log_scale]*len(indices)

        for k, param_index in enumerate(indices):
            print(
                "\n Profile likelihood for " + param_names[param_index] + ":"
            )
            self.timer.reset()

            if log_scale[k]:
                proportion = ((
                        np.log(self.optimum_point[param_index]) -
                        np.log(self.bounds[0, param_index])
                    )/(
                        np.log(self.bounds[1, param_index]) -
                        np.log(self.bounds[0, param_index])
                ))
                evals_lower = max(min(
                    int(tot_evals*proportion), tot_evals-1
                ), 1)
                range_lower = np.logspace(
                    np.log(self.optimum_point[param_index]),
                    0.75*np.log(self.optimum_point[param_index]) + 0.25*np.log(self.bounds[0, param_index]),
                    evals_lower,
                    base=np.e
                )
                range_upper = np.logspace(
                    np.log(self.optimum_point[param_index]),
                    0.75*np.log(self.optimum_point[param_index]) + 0.25*np.log(self.bounds[1, param_index]),
                    tot_evals-evals_lower,
                    base=np.e
                )

            else:
                range_lower = np.linspace(
                    self.optimum_point[param_index],
                    max(
                        0.5*self.optimum_point[param_index],
                        self.bounds[0, param_index]
                    ),
                    int(tot_evals/2.0)
                )
                range_upper = np.linspace(
                    self.optimum_point[param_index],
                    min(
                        1.5*self.optimum_point[param_index],
                        self.bounds[1, param_index]
                    ),
                    tot_evals - int(tot_evals/2.0)
                )

            self._n = 0
            likelihood_lower = self.single_prof_ll(param_index, range_lower)

            self._n = 0.5
            likelihood_upper = self.single_prof_ll(param_index, range_upper)

            likelihood_lower = np.flipud(likelihood_lower)
            likelihood = np.concatenate((likelihood_lower, likelihood_upper))
            np.save(save_file+log_scale[k]*"_log"+str(param_index), likelihood)

    def save_pairwise_profiles(
        self,
        save_file,
        tot_evals,
        indices=None,
        param_names=None,
        log_scale=False
    ):
        self.per_done = 0
        if param_names is None:
            param_names = np.char.add(
                np.asarray(["Parameter "]*len(self.optimum_point)),
                np.arange(0, len(self.optimum_point)).astype(str)
            )
        if indices is None:
            indices = (
                range(0, len(self.optimum_point)),
                range(0, len(self.optimum_point))
            )
        if type(log_scale) is bool:
            log_scale = [log_scale]*len(indices)

        for index_1 in indices[0]:

            if log_scale[index_1]:
                proportion = ((
                        np.log(self.optimum_point[index_1]) -
                        np.log(self.bounds[0, index_1])
                    )/(
                        np.log(self.bounds[1, index_1]) -
                        np.log(self.bounds[0, index_1])
                ))
#                 evals_mid = int(0.5*tot_evals)
                evals_lower = max(min(
                    int(tot_evals*proportion), tot_evals-1
                ), 1)
                evals_upper = tot_evals-evals_lower
                
                range_1_l = np.logspace(
                    np.log(self.optimum_point[index_1]),
                    0.25*np.log(self.bounds[0, index_1])+0.75*np.log(self.optimum_point[index_1]),
                    evals_lower,
                    base=np.e
                )
                
                range_1_u = np.logspace(
                    np.log(self.optimum_point[index_1]),
                    0.75*np.log(self.optimum_point[index_1]) + 0.25*np.log(self.bounds[1, index_1]),
                    evals_upper,
                    base=np.e
                )
            else:
                range_1_l = np.linspace(
                    self.optimum_point[index_1],
                    max(
                        0.5*self.optimum_point[index_1],
                        self.bounds[0, index_1]
                    ),
                    int(tot_evals/2.0)
                )
                range_1_u = np.linspace(
                    self.optimum_point[index_1],
                    min(
                        1.5*self.optimum_point[index_1],
                        self.bounds[1, index_1]
                    ),
                    tot_evals - int(tot_evals/2.0)
                )

            y_range = np.concatenate((range_1_l[::-1], range_1_u))

            for index_2 in filter(lambda x: x < index_1, indices[1]):
                print(
                    "\n Pairwise Profile likelihood for " +
                    param_names[index_1] +
                    " and " + param_names[index_2] + ":"
                )
                self.timer.reset()

                if log_scale[index_2]:
                    proportion = ((
                            np.log(self.optimum_point[index_2]) -
                            np.log(self.bounds[0, index_2])
                        )/(
                            np.log(self.bounds[1, index_2]) -
                            np.log(self.bounds[0, index_2])
                    ))
                    
                    evals_left = max(min(
                        int((tot_evals)*proportion), tot_evals-1
                    ), 1)
                    evals_right = tot_evals-evals_lower
                    
                    range_2_l = np.logspace(
                        np.log(self.optimum_point[index_2]),
                        np.log(self.bounds[0, index_2]),
                        evals_left,
                        base=np.e
                    )
                    range_2_r = np.logspace(
                        np.log(self.optimum_point[index_2]),
                        np.log(self.bounds[1, index_2]),
                        evals_right,
                        base=np.e
                    )

                else:
                    range_2_l = np.linspace(
                        self.optimum_point[index_2],
                        max(
                            0.5*self.optimum_point[index_2],
                            self.bounds[0, index_2]
                        ),
                        int(tot_evals/2.0)
                    )
                    range_2_r = np.linspace(
                        self.optimum_point[index_2],
                        min(
                            1.5*self.optimum_point[index_2],
                            self.bounds[1, index_2]
                        ),
                        tot_evals - int(tot_evals/2.0)
                    )

                self._n = 0
                ll_lowleft = self.pair_prof_ll(
                    (index_1, index_2),
                    [range_1_l, range_2_l]
                )
                ll_lowleft = np.fliplr(ll_lowleft[:, 1:])

                self._n = 0.25
                ll_upleft = self.pair_prof_ll(
                    (index_1, index_2),
                    [range_1_u, range_2_l]
                )
                ll_upleft = np.fliplr(np.flipud(ll_upleft[1:, 1:]))

                self._n = 0.5
                ll_lowright = self.pair_prof_ll(
                    (index_1, index_2),
                    [range_1_l, range_2_r]
                )

                self._n = 0.75
                ll_upright = self.pair_prof_ll(
                    (index_1, index_2),
                    [range_1_u, range_2_r]
                )
                ll_upright = np.flipud(ll_upright[1:, :])

                ll_upper = np.concatenate((ll_upleft, ll_upright), axis=1)
                ll_lower = np.concatenate((ll_lowleft, ll_lowright), axis=1)
                likelihood = np.concatenate((ll_upper, ll_lower), axis=0)

                x_range = np.concatenate((range_2_l[::-1], range_2_r))

                np.save((
                    save_file+"_" +
                    log_scale[index_1]*"log"+str(index_1) + "_" +
                    log_scale[index_2]*"log"+str(index_2)
                ), (likelihood, x_range, y_range))


if __name__ == '__main__':
    drug = 'Docetaxel'
    num_comp = 2
    observation_name = 'Platelets '
    PD_actual_params = np.load(
        "./Data_and_parameters/PD_sim/actual_params.npy"
    )
    PD_param_names = PD_actual_params[0]
    PD_actual_params = PD_actual_params[1]
    PD_actual_params = PD_actual_params.astype('float64')

    PK_params = np.load("./Data_and_parameters/PK_sim/actual_params.npy")[1, :]
    PK_params = PK_params.astype('float64')
    df = pandas.read_csv(
        "./Data_and_parameters/PD_sim/sythesised_data_real_timepoints.csv"
    )
    df_before_0 = df[df["TIME"] < 0]
    R_0_approx = np.mean(df_before_0["OBS"])

    opt_df = pandas.read_csv("./Data_and_parameters/PD_sim/opt_results.csv")
    opt_point_comb = np.asarray(opt_df['Combined Noise'][0:-1])

    lower_bound = [0.1*R_0_approx, df['TIME'].max()*0.01, 0.005, 0.01,     0.001,      0.001, 0.001]
    upper_bound = [10*R_0_approx,      df['TIME'].max(),     5,    100,   R_0_approx,   10,    0.99]

    PD_model = Pints_PD_model(PK_params[:-1], data=df, num_comp=2)
    problem = pints.SingleOutputProblem(
        PD_model, PD_model.pseudotime, df['OBS'].to_numpy()
    )
    ll = ConstantAndMultiplicativeGaussianLogLikelihood(problem)
    profile_ll = profile_loglikelihood(
        ll,
        opt_point_comb,
        np.asarray([lower_bound, upper_bound])
    )

    # For double parameters
    profile_ll.save_pairwise_profiles(
        "./Data_and_parameters/PD_sim/identify_comb_param",
        20,
        log_scale=[False]*4+[True]*3,
        param_names=PD_param_names,
        indices=[[2], [1]]
    )
