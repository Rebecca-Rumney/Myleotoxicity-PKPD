import numpy as np
import pints


class GaussianLogLikelihood(pints.ProblemLogLikelihood):
    def __init__(self, problem):
        super(GaussianLogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * np.log(2 * np.pi)

        # Set up pointwise loglikelihoods
        self._last_pointwise_loglikelihoods = None

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        error = self._values - self._problem.evaluate(x[:-self._no])
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))

    def create_pointwise_loglikelihoods(self, parameters):
        """
        Returns a matrix of size nt x no containing the loglikelihood of each
        observation and at each time point with the given parameters
        """
        sigma = np.asarray(parameters[-self._no:])
        error = self._values - self._problem.evaluate(parameters[:-self._no])
        point_loglike = (
            -0.5 * np.log(2 * np.pi) -
            np.log(sigma) -
            error**2 / (2 * sigma**2)
            )
        return point_loglike

    def get_last_pointwise_loglikelihoods(self):
        return self._last_pointwise_loglikelihoods

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        sigma = np.asarray(x[-self._no:])

        # Evaluate, and get residuals
        y, dy = self._problem.evaluateS1(x[:-self._no])

        # Reshape dy, in case we're working with a single-output problem
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)

        # Note: Must be (data - simulation), sign now matters!
        r = self._values - y

        # Calculate log-likelihood
        L = self.__call__(x)

        # Calculate derivatives in the model parameters
        dL = np.sum(
            (sigma**(-2.0) * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)

        # Calculate derivative wrt sigma
        dsigma = -self._nt / sigma + sigma**(-3.0) * np.sum(r**2, axis=0)
        dL = np.concatenate((dL, np.array(list(dsigma))))

        # Return
        return L, dL


class MultiplicativeGaussianLogLikelihood(pints.ProblemLogLikelihood):

    def __init__(self, problem, fix_eta=None):
        super(MultiplicativeGaussianLogLikelihood, self).__init__(problem)

        # Get number of times and number of outputs
        self._nt = len(self._times)
        no = problem.n_outputs()
        self._fix_eta = fix_eta
        if self._fix_eta is None:
            self._np = 2 * no  # 2 parameters added per output
        else:
            self._np = no  # 1 parameters added per output

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = 0.5 * np.log(2 * np.pi)

    def __call__(self, x):
        pointwise = self.create_pointwise_loglikelihoods(x)
        self._last_pointwise_loglikelihoods = pointwise
        return np.sum(pointwise)

    def create_pointwise_loglikelihoods(self, parameters):
        """
        Returns a matrix of size nt x no containing the loglikelihood of each
        observation and at each time point with the given parameters
        """
        noise_parameters = np.asarray(parameters[-self._np:])
        if self._fix_eta is None:
            eta = np.asarray(noise_parameters[0::2])
            sigma = np.asarray(noise_parameters[1::2])
        else:
            sigma = noise_parameters
            eta = self._fix_eta

#         Evaluate function (n_times, n_output)
        function_values = self._problem.evaluate(parameters[:-self._np])
        error = self._values - function_values
        noise_term = function_values**eta * sigma
        point_loglike = (
            -0.5 * np.log(2 * np.pi) -
            np.log(noise_term) -
            error**2 / (2 * noise_term**2)
            )
        return point_loglike

    def get_last_pointwise_loglikelihoods(self):
        return self._last_pointwise_loglikelihoods

#     def __call__(self, x):
#         # Get noise parameters
#         noise_parameters = x[-self._np:]
#         eta = np.asarray(noise_parameters[0::2])
#         sigma = np.asarray(noise_parameters[1::2])

#         # Evaluate function (n_times, n_output)
#         function_values = self._problem.evaluate(x[:-self._np])

#         # Compute likelihood
#         log_likelihood = \
#             -self._logn - np.sum(
#                 np.sum(np.log(function_values**eta * sigma), axis=0)
#                 + 0.5 / sigma**2 * np.sum(
#                     (self._values - function_values)**2
#                     / function_values ** (2 * eta), axis=0))

#         return log_likelihood


class MultiplicativeGaussianLogLikelihoodFixEta(pints.ProblemLogLikelihood):

    def __init__(self, problem):
        super(
            MultiplicativeGaussianLogLikelihoodFixEta, self
            ).__init__(problem)

        # Get number of times and number of outputs
        self._nt = len(self._times)
        no = problem.n_outputs()
        self._np = 1  # 1 parameters added

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = 0.5 * self._nt * no * np.log(2 * np.pi)

    def __call__(self, x):
        # Get noise parameters
        noise_parameters = x[-self._np:]
        eta = 1  # np.asarray(noise_parameters[0])
        sigma = np.asarray(noise_parameters[0])

        # Evaluate function (n_times, n_output)
        function_values = self._problem.evaluate(x[:-self._np])

        # Compute likelihood
        log_likelihood = \
            -self._logn - np.sum(
                np.sum(np.log(function_values**eta * sigma), axis=0)
                + 0.5 / sigma**2 * np.sum(
                    (self._values - function_values)**2
                    / function_values ** (2 * eta), axis=0))

        return log_likelihood

    def pointwiseLogliklihoods(self, parameters):
        # Get parameters from input
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[2 * self._no:]

        # Evaluate noise-free model (n_times, n_outputs)
        function_values = self._problem.evaluate(parameters[:-self._np])

        # Compute error (n_times, n_outputs)
        error = self._values - function_values

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * function_values**eta

        # Compute log-likelihood
        # (inner sums over time points, outer sum over parameters)
        log_likelihood = self._logn - np.sum(
            np.sum(np.log(sigma_tot), axis=0)
            + 0.5 * np.sum(error**2 / sigma_tot**2, axis=0))

        return log_likelihood


class ConstantAndMultiplicativeGaussianLogLikelihood(
        pints.ProblemLogLikelihood):
    def __init__(self, problem, fix_noise=None):
        super(ConstantAndMultiplicativeGaussianLogLikelihood, self).__init__(
            problem)

        # Get number of times and number of noise parameters
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        if fix_noise is None:
            self._fix_param = [None]*3
        else:
            self._fix_param = fix_noise
        self._np = (self._fix_param.count(None)) * self._no

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = -0.5 * np.log(2 * np.pi)

    def __call__(self, x):
        pointwise = self.create_pointwise_loglikelihoods(x)
        self._last_pointwise_loglikelihoods = pointwise
        return np.sum(pointwise)

    def create_pointwise_loglikelihoods(self, parameters):
        """
        Returns a matrix of size nt x no containing the loglikelihood of each
        observation and at each time point with the given parameters
        """
        if self._np == 0:
            function_values = self._problem.evaluate(parameters)
            noise_parameters = np.asarray([])
        else:
            function_values = self._problem.evaluate(parameters[:-self._np])
            noise_parameters = np.asarray(parameters[-self._np:])
        error = self._values - function_values
        
        no_used = 0
        if self._fix_param[0] is None:
            sigma_base = noise_parameters[no_used: no_used + self._no]
            no_used = no_used + self._no
        else:
            sigma_base = np.array([self._fix_param[0]] * self._no)
        if self._fix_param[1] is None:
            eta = noise_parameters[no_used: no_used + self._no]
            no_used = no_used + self._no
        else:
            eta = np.array([self._fix_param[1]] * self._no)
        if self._fix_param[2] is None:
            sigma_rel = noise_parameters[no_used: no_used + self._no]
            no_used = no_used + self._no
        else:
            sigma_rel = np.array([self._fix_param[2]] * self._no)

        sigma_tot = sigma_base + sigma_rel * function_values**eta
        point_loglike = (
            -0.5 * np.log(2 * np.pi) -
            np.log(sigma_tot) -
            0.5 * error**2 / (sigma_tot**2)
            )
        return point_loglike

    def get_last_pointwise_loglikelihoods(self):
        return self._last_pointwise_loglikelihoods

#     def __call__(self, parameters):
#         # Get parameters from input
#         noise_parameters = np.asarray(parameters[-self._np:])
#         sigma_base = noise_parameters[:self._no]
#         eta = noise_parameters[self._no:2 * self._no]
#         sigma_rel = noise_parameters[2 * self._no:]

#         # Evaluate noise-free model (n_times, n_outputs)
#         function_values = self._problem.evaluate(parameters[:-self._np])

#         # Compute error (n_times, n_outputs)
#         error = self._values - function_values

#         # Compute total standard deviation
#         sigma_tot = sigma_base + sigma_rel * function_values**eta

#         # Compute log-likelihood
#         # (inner sums over time points, outer sum over parameters)
#         log_likelihood = self._logn - np.sum(
#             np.sum(np.log(sigma_tot), axis=0)
#             + 0.5 * np.sum(error**2 / sigma_tot**2, axis=0))

#         return log_likelihood

    def evaluateS1(self, parameters):
        # Get parameters from input
        # Shape sigma_base, eta, sigma_rel = (n_outputs,)
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[-self._no:]

        # Evaluate noise-free model, and get residuals
        # y shape = (n_times,) or (n_times, n_outputs)
        # dy shape = (n_times, n_model_parameters) or
        # (n_times, n_outputs, n_model_parameters)
        y, dy = self._problem.evaluateS1(parameters[:-self._np])

        # Reshape y and dy, in case we're working with a single-output problem
        # Shape y = (n_times, n_outputs)
        # Shape dy = (n_model_parameters, n_times, n_outputs)
        y = y.reshape(self._nt, self._no)
        dy = np.transpose(
            dy.reshape(self._nt, self._no, self._n_parameters - self._np),
            axes=(2, 0, 1))

        # Compute error
        # Note: Must be (data - simulation), sign now matters!
        # Shape: (n_times, output)
        error = self._values.reshape(self._nt, self._no) - y

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * y**eta

        # Compute likelihood
        L = self.__call__(parameters)

        # Compute derivative w.r.t. model parameters
        dtheta = -np.sum(sigma_rel * eta * np.sum(
            y**(eta - 1) * dy / sigma_tot, axis=1), axis=1) + \
            np.sum(error * dy / sigma_tot**2, axis=(1, 2)) + np.sum(
                sigma_rel * eta * np.sum(
                    error**2 * y**(eta - 1) * dy / sigma_tot**3, axis=1),
                axis=1)

        # Compute derivative w.r.t. sigma base
        dsigma_base = - np.sum(1 / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3, axis=0)

        # Compute derivative w.r.t. eta
        deta = -sigma_rel * (
            np.sum(y**eta * np.log(y) / sigma_tot, axis=0) -
            np.sum(
                error**2 / sigma_tot**3 * y**eta * np.log(y),
                axis=0))

        # Compute derivative w.r.t. sigma rel
        dsigma_rel = -np.sum(y**eta / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3 * y**eta, axis=0)

        # Collect partial derivatives
        dL = np.hstack((dtheta, dsigma_base, deta, dsigma_rel))

        # Return
        return L, dL


class ConstantAndMultiplicativeGaussianLogLikelihoodFixEta(
        pints.ProblemLogLikelihood):

    def __init__(self, problem):
        super(
            ConstantAndMultiplicativeGaussianLogLikelihoodFixEta, self
            ).__init__(problem)

        # Get number of times and number of noise parameters
        self._nt = len(self._times)
        self._no = problem.n_outputs()
        self._np = 2 * self._no

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = -0.5 * self._nt * self._no * np.log(2 * np.pi)

    def __call__(self, parameters):
        # Get parameters from input
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = 1
        sigma_rel = noise_parameters[self._no:]

        # Evaluate noise-free model (n_times, n_outputs)
        function_values = self._problem.evaluate(parameters[:-self._np])

        # Compute error (n_times, n_outputs)
        error = self._values - function_values

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * function_values**eta

        # Compute log-likelihood
        # (inner sums over time points, outer sum over parameters)
        log_likelihood = self._logn - np.sum(
            np.sum(np.log(sigma_tot), axis=0)
            + 0.5 * np.sum(error**2 / sigma_tot**2, axis=0))

        return log_likelihood

    def evaluateS1(self, parameters):
        # Get parameters from input
        # Shape sigma_base, eta, sigma_rel = (n_outputs,)
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = 1
        sigma_rel = noise_parameters[self._no:]

        # Evaluate noise-free model, and get residuals
        # y shape = (n_times,) or (n_times, n_outputs)
        # dy shape = (n_times, n_model_parameters) or
        # (n_times, n_outputs, n_model_parameters)
        y, dy = self._problem.evaluateS1(parameters[:-self._np])

        # Reshape y and dy, in case we're working with a single-output problem
        # Shape y = (n_times, n_outputs)
        # Shape dy = (n_model_parameters, n_times, n_outputs)
        y = y.reshape(self._nt, self._no)
        dy = np.transpose(
            dy.reshape(self._nt, self._no, self._n_parameters - self._np),
            axes=(2, 0, 1))

        # Compute error
        # Note: Must be (data - simulation), sign now matters!
        # Shape: (n_times, output)
        error = self._values.reshape(self._nt, self._no) - y

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * y**eta

        # Compute likelihood
        L = self.__call__(parameters)

        # Compute derivative w.r.t. model parameters
        dtheta = -np.sum(sigma_rel * eta * np.sum(
            y**(eta - 1) * dy / sigma_tot, axis=1), axis=1) + \
            np.sum(error * dy / sigma_tot**2, axis=(1, 2)) + np.sum(
                sigma_rel * eta * np.sum(
                    error**2 * y**(eta - 1) * dy / sigma_tot**3, axis=1),
                axis=1)

        # Compute derivative w.r.t. sigma base
        dsigma_base = - np.sum(1 / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3, axis=0)

        # Compute derivative w.r.t. eta
        deta = -sigma_rel * (
            np.sum(y**eta * np.log(y) / sigma_tot, axis=0) -
            np.sum(
                error**2 / sigma_tot**3 * y**eta * np.log(y),
                axis=0))

        # Compute derivative w.r.t. sigma rel
        dsigma_rel = -np.sum(y**eta / sigma_tot, axis=0) + np.sum(
            error**2 / sigma_tot**3 * y**eta, axis=0)

        # Collect partial derivatives
        dL = np.hstack((dtheta, dsigma_base, deta, dsigma_rel))

        # Return
        return L, dL
