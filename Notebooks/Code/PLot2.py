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

        self.default_colour = {
            "base": "rebeccapurple", "individual": "viridis",
            "zero dose": "turbo"
        }

        self.base_colour = self.default_colour["base"]
        self.ind_colour_scale = self.default_colour["individual"]
        self.ind_0_colour_scale = self.default_colour["zero dose"]

        if data is None:
            self.data_set = False
            self.n_ind = 0
        else:
            self.set_data(data)
            self.n_ind = np.sum(self.n_ids_per_dose)
        self.pop_model = pop_model
        self.mech_model = mech_model
        self.error_model = error_models
        self.prior_model = prior_model

        if self.pop_model is None:
            # TODO: How to tell number of ind params when there is no pop_model
            self.n_ind_params = 0
        else:
            if not self.data_set:
                self.n_ind = self.pop_model.n_ids()
            self.n_ind_params = self.n_ind*self.pop_model.n_hierarchical_dim()

    def set_colour(self, colour_scheme):

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
        if "zero dose individual" in colour_scheme.keys():
            self.ind_0_colour_scale = colour_scheme["zero dose"]
            if self.ind_0_colour_scale is None:
                self.ind_0_colour_scale = self.default_colour["zero dose"]

        if self.data_set:
            # Set up the colour scheme for individuals
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

    def optimise(self, function, start, fix=None, minimise=False):

        # PINTS optimisers minimise.
        # To maximise the function, f, we will need to minimise -f
        if minimise:
            sign = 1
        else:
            sign = -1

        if fix is not None:
            fix = np.asarray(fix)
            if len(fix.shape) == 1:
                delete_arg = int(fix[0])
                insert_arg = int(fix[0])
                fix_values = fix[1]
            else:
                fix = fix[:, np.argsort(fix[0])]
                insert_arg = fix[0].astype(int)-np.arange(len(fix[0]))
                delete_arg = fix[0].astype(int)
                fix_values = fix[1]

            opt_start = np.delete(start, delete_arg)

            def minimise_func(reduced_params):
                full_params = np.insert(reduced_params, insert_arg, fix_values)
                return sign*function(full_params)
        else:
            def minimise_func(full_params):
                return sign*function(full_params)
        optimiser = pints.NelderMead(opt_start)
        significant_change = 1e-4
        max_iters = 10000
        max_unchanged_iters = 200

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

        if fix is not None:
            xbest = np.insert(optimiser.xbest(), insert_arg, fix_values)
        else:
            xbest = optimiser.xbest()

        return xbest, sign*fbest
    
    def f_over_param_range(
            self, f, i_param, param_range, params_ref, pairwise=False, normalise=True,
            individual_parameters=False, n_evals=50, record_param_opt=False
    ):
        if pairwise:
            pass
        else:
            x_values = np.linspace(param_range[0], param_range[1], n_evals)

            curr = params_ref.copy()

            # Start from the centre, useful for profile likelihoods
            lower_incl = x_values <= params_ref[i_param+self.n_ind_params]
            x_values_lower = x_values[lower_incl][::-1]
            x_values_upper = x_values[np.logical_not(lower_incl)]
            x_values = np.concatenate((x_values_lower, x_values_upper))
            result = np.empty_like(x_values)
            param_values = np.empty((len(x_values), len(curr)))
            for i_x, x in enumerate(x_values):
                if i_x == len(x_values_lower):
                    # Reset start point for second half
                    curr = params_ref.copy()
                curr, result[i_x] = f(x, i_param+self.n_ind_params, curr=curr)
                param_values[i_x] = curr
            max_score = np.max(result)

            # Sort the results for plotting
            sort = np.argsort(x_values)
            result = result[sort] - max_score  # Normalise the result
            x_values = x_values[sort]
            param_values = param_values[sort]

            return x_values, result, param_values

    def create_function_for_plotting(self, function, params_ref, profile=None):

        def slice_function(param_value, param_arg, curr=None):
            slice_param = params_ref.copy()
            slice_param[param_arg] = param_value
            score = function(slice_param)
            return slice_param, score

        if profile is not None:
            def profile_function(param_value, param_arg, curr):
                minimise = profile == "minimum"
                opt_param, score = self.optimise(
                    function,
                    curr,
                    fix=(param_arg, param_value),
                    minimise=minimise
                )
                return opt_param, score

            profile = True
            f = profile_function
        else:
            profile = False
            f = slice_function
        return f

    def plot_param_function(
            self, function, params_ref, profile=None, pairwise=False,
            individual_parameters=False, param_names=None, bounds=(None, None),
            force_bounds=(False, False), n_evals=50,
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
        slice_function = self.create_function_for_plotting(function, params_ref, profile=None)

        if profile is not None:
            f = self.create_function_for_plotting(function, params_ref, profile=profile)
            profile = True
        else:
            f = slice_function
            profile = False
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
        if pairwise:
            pair_colour = "dense"

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
            if not force_lower_bounds[i_param]:
                while i_check == 0 and j_check <= 20:
                    x_check = np.linspace(param_ref_i, x_min, 20)[1:]
                    for x in x_check:
                        _, score = slice_function(x, i_param+self.n_ind_params)
                        if score < (ref_score - 3*1.92):
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
                        if score < (ref_score - 3*1.92):
                            x_max = x_check[i_check]
                            break
                        i_check += 1
                    j_check += 1
            param_ranges[i_param] = (x_min, x_max)

            # Calculate function
            x_values, result, _ = self.f_over_param_range(f, i_param, param_ranges[i_param], params_ref, n_evals=n_evals)
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
                                if line_start and profile:
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
                            colorscale=pair_colour,
                            zmin=- 1.92,
                            zmax=+ 1.92
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
    
    def plot_ind_profile_ll(
            self, log_likelihood, i_param, params_ref,
            param_name=None, bounds=(None, None),
            force_bounds=(False, False), n_evals=50,
    ):
        # Create function
        slice_function = self.create_function_for_plotting(log_likelihood, params_ref, profile=None)
        f = self.create_function_for_plotting(log_likelihood, params_ref, profile="maximise")
        ref_score = log_likelihood(params_ref)
        param_ref_i = params_ref[i_param+self.n_ind_params]

        # Set name if not provided
        if (param_name is None) and (self.pop_model is not None):
            param_name = self.pop_model.get_parameter_names()[i_param]
        elif param_name is None:
            param_name = 'Parameter ' + str(i_param+1)

        # Create subplots
        n_rows = 3
        n_cols = 1
        fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True)

        # Set up colours
        pop_colour = 'darkgrey'
        if self.data_set:
            ind_colour_selection = self.ind_colours
        else:
            ind_colour_selection = [self.base_colour]*self.n_ind

        # Set up param range
        if bounds[0] is None:
            lower_bound = 0.5*param_ref_i
        else:
            lower_bound = bounds[0]
        if bounds[1] is None:
            upper_bound = 1.5*param_ref_i
        else:
            upper_bound = bounds[1]
        
        x_min = lower_bound
        i_check = 0
        j_check = 0
        if not force_bounds[0]:
            while i_check == 0 and j_check <= 20:
                x_check = np.linspace(param_ref_i, x_min, 20)[1:]
                for x in x_check:
                    _, score = slice_function(x, i_param+self.n_ind_params)
                    if score < (ref_score - 6*1.92):
                        x_min = x_check[i_check]
                        break
                    i_check += 1
                j_check += 1
        x_max = upper_bound
        i_check = 0
        j_check = 0
        if not force_bounds[1]:
            while i_check == 0 and j_check <= 20:
                x_check = np.linspace(param_ref_i, x_max, 20)[1:]
                for x in x_check:
                    _, score = slice_function(x, i_param+self.n_ind_params)
                    if score < (ref_score - 6*1.92):
                        x_max = x_check[i_check]
                        break
                    i_check += 1
                j_check += 1
        param_range = (x_min, x_max)

        # Calculate function
        x_values, result, params_result = self.f_over_param_range(f, i_param, param_range, params_ref, n_evals=n_evals)
        fig.add_trace(
            go.Scatter(
                name='Population Log-likelihood',
                x=x_values,
                y=result,
                mode='lines',
                line=dict(color=pop_colour),
            ),
            row=1,
            col=1
        )
        # create matrix of pointwise loglikelihoods of shape (n_x_values, n_ind)
        ind_ll = np.empty((x_values.shape[0], self.n_ind))
        for i_vec, param_vec in enumerate(params_result):
            if param_vec[i_param+self.n_ind_params] != x_values[i_vec]:
                raise ValueError(
                    "Incorrect param vector provided for "
                    + str(i_vec) +"th parameter vector:"
                    + str(param_vec[i_param+self.n_ind_params]) + "!=" + str(x_values[i_vec])
                )
            pll = np.array(log_likelihood.compute_pointwise_ll(param_vec, per_individual=True))
            full_ll = log_likelihood(param_vec)
            if np.abs(np.sum(pll) - full_ll) > 1e-3:
                raise ValueError(
                    "Incorrect pointwise log-likelihood calculated for "
                    + str(i_vec) +"th parameter vector:"
                    + str(np.sum(pll)) + "!=" + str(full_ll)
                )
            if pll.shape != (self.n_ind, ):
                raise ValueError(
                    "individual log-likelihood is wrong shape. Should be "
                    + str((self.n_ind, ))
                    + "but is " + str(pll.shape)
                )
            ind_ll[i_vec] = pll
        result_from_ind_ll = np.sum(ind_ll, axis=1)
        max_result = np.max(result_from_ind_ll)
        result_from_ind_ll -= max_result
        check_ind_ll = np.abs(result_from_ind_ll - result) > 1e-2
        if np.any(check_ind_ll):
            raise ValueError(
                "Incorrect profile log-likelihood calculated for "
                + str(np.argwhere(check_ind_ll)) +"th parameter vectors:"
                + str(ind_ll[check_ind_ll])
            )
        ind_ll = ind_ll - np.array(log_likelihood.compute_pointwise_ll(params_ref, per_individual=True)) # np.max(ind_ll, axis=0)  # (max_result/self.n_ind)
        for i_ind in range(0, self.n_ind):
            ind_colour = ind_colour_selection.flatten()[i_ind]
            fig.add_trace(
                go.Scatter(
                    name='Individual Log-likelihoods',
                    x=x_values,
                    y=ind_ll[:, i_ind],
                    mode='lines',
                    line=dict(color=ind_colour, width=1),
                    showlegend=False
                ),
                row=2,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    name='Individual Parameter optimised',
                    x=x_values,
                    y=params_result[:, i_ind]-params_ref[i_ind],
                    mode='lines',
                    line=dict(color=ind_colour, width=1),
                    showlegend=False
                ),
                row=3,
                col=1
            )
        
        pop_model = log_likelihood.get_population_model()
        sp_dims = pop_model.get_special_dims()[0]

        n_params = pop_model.n_parameters()
        non_mix_params = np.asarray([range(x, y) for _, _, x, y, _ in sp_dims]).flatten()
        mix_params = [x + self.n_ind for x in range(0, n_params) if x not in non_mix_params]
        for typ_param in mix_params[::2]:
            fig.add_trace(
                go.Scatter(
                    name='Population Parameter optimised',
                    x=x_values,
                    y=np.exp(params_result[:, typ_param])-np.exp(params_ref[typ_param]),
                    mode='lines',
                    line = dict(color=pop_colour),
                    showlegend=False
                ),
                row=3,
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
