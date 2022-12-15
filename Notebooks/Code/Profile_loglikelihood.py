import pandas
from Code.PD_model import PintsPDFribergLinE as Pints_PD_model
from Code.Likelihoods import ConstantAndMultiplicativeGaussianLogLikelihood
import pints
import nlopt
import numpy as np


class profile_loglikelihood():
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
#                 print(range_1_l[::-1], range_1_u)
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
