# WCEI acquisition function class
from Packages import *


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""
    Base class for analytic acquisition functions.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        if posterior_transform is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a "
                    "multi-output model."
                )
        else:
            if not isinstance(posterior_transform, PosteriorTransform):
                raise UnsupportedError(
                    "AnalyticAcquisitionFunctions only support PosteriorTransforms."
                )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )


    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the first and second moments of the model posterior.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            compute_sigma: Boolean indicating whether or not to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        self.to(device=X.device)  # ensures buffers / parameters are on the same device
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma
    

class WeightedConstrainedEI(AnalyticAcquisitionFunction):
    r""" Weighted Constrained Expected Improvement """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        best_c: Union[float, Tensor],
        raw_samples,
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        weight_samples,
        maximize: bool = True,
    ) -> None:
        r"""Weighted Constrained Expected Improvement.

        Args:
            model: A fitted multi-output model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the current utility function value observed so far (assumed noiseless).
            best_c: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the current constrain function value observed so far (assumed noiseless).
            raw_samples: number of weight samples (r)
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            weight_samples: r x m Tensor of weight samples
            maximize: If True, consider the problem a maximization problem.
        """
        # Use AcquisitionFunction constructor to avoid check for posterior transform.
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.constraints = constraints
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_c", torch.as_tensor(best_c))
        self.raw_samples = raw_samples
        self.weight_samples = weight_samples
        _preprocess_weighted_constraint_bounds(self, constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        ind = self.objective_index
        mean_obj, sigma_obj = means[..., ind], sigmas[..., ind]
        res = torch.zeros_like(mean_obj) # (b)
        prob_feas = _compute_weighted_prob_feas(self, means=means, sigmas=sigmas) # (b) x n+1
        for i in range(len(self.best_f)):
            u = _scaled_improvement(mean_obj, sigma_obj, self.best_f[i], self.maximize)
            if self.best_f[i] == self.best_f.min():
                ei = torch.zeros_like(u) + 1
            else:
                ei = sigma_obj * _ei_helper(u)
            res = res + ei.mul(prob_feas[...,i])

        return res



def _preprocess_weighted_constraint_bounds(
    acqf: WeightedConstrainedEI,
    constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
) -> None:
    r"""Set up constraint bounds.

    Args:
        constraints: A dictionary of the form `{i: [lower, upper]}`, where
            `i` is the output index, and `lower` and `upper` are lower and upper
            bounds on that output (resp. interpreted as -Inf / Inf if None)
    """
    con_lower, con_lower_inds = [], []
    con_upper, con_upper_inds = [], []
    con_both, con_both_inds = [], []
    con_indices = list(constraints.keys())
    if len(con_indices) == 0:
        raise ValueError("There must be at least one constraint.")
    if acqf.objective_index in con_indices:
        raise ValueError(
            "Output corresponding to objective should not be a constraint."
        )
    for k in con_indices:
        if constraints[k][0] is not None and constraints[k][1] is not None:
            if constraints[k][1] <= constraints[k][0]:
                raise ValueError("Upper bound is less than the lower bound.")
            con_both_inds.append(k)
            con_both.append([constraints[k][0], constraints[k][1]])
        elif constraints[k][0] is not None:
            con_lower_inds.append(k)
            con_lower.append(constraints[k][0])
        elif constraints[k][1] is not None:
            con_upper_inds.append(k)
            con_upper.append(constraints[k][1])
    # tensor-based indexing is much faster than list-based advanced indexing
    for name, indices in [
        ("con_lower_inds", con_lower_inds),
        ("con_upper_inds", con_upper_inds),
        ("con_both_inds", con_both_inds),
        ("con_both", con_both),
        ("con_lower", con_lower),
        ("con_upper", con_upper)
    ]:
        acqf.register_buffer(name, tensor=torch.as_tensor(indices))



def _compute_weighted_prob_feas(
    acqf: WeightedConstrainedEI,
    means: Tensor,
    sigmas: Tensor,
) -> Tensor:
    r"""Compute weighted logarithm of the feasibility probability for each batch of X.

    Args:
        X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
            points each.
        means: A `(b) x 1+m`-dim Tensor of means.
        sigmas: A `(b) x 1+m`-dim Tensor of standard deviations.
    Returns:
        A `b`-dim tensor of log feasibility probabilities

    Note: This function does case-work for upper bound, lower bound, and both-sided
    bounds. Another way to do it would be to use 'inf' and -'inf' for the
    one-sided bounds and use the logic for the both-sided case. But this
    causes an issue with autograd since we get 0 * inf.
    Note2:
        r = acqf.raw_samples
        m = means.shape[1]-1
        n = len(acqf.best_f)-1
        b = means.shape[0]
    """
    weight_samples = acqf.weight_samples # r x 1

    acqf.to(device=means.device)
    # prob: int_Aj P(cond(a))p(a), j=1:n+1 for each of the b batches (b x n+1 tensor)
    prob = torch.ones(means.shape[0],len(acqf.best_f)) # b x n+1
    idx = 0 # count for s_w (0 -> m-1)
    if len(acqf.con_lower_inds) > 0: # cond: c > a
        # dist_l: r samples of P(c > a) for each of the b batches (r x b tensor)
        dist_l = torch.ones(acqf.raw_samples, means[..., 0].shape[0])
        for i in acqf.con_lower_inds:
            # turn Gaussian samples into samples from p(a)
            ws = weight_samples[...,idx].repeat(means[..., 0].shape[0], 1).T # r x b

            # compute P(c_i > a_i) and multiply it to dist_l
            dist_l *= Phi(-(ws + acqf.con_lower[idx] - means[..., i]) / sigmas[..., i]) # r x b
            idx += 1 # increase s_w count

        # get indices for each Ai
        idx_rest = Ai_index(acqf.best_c,0,weight_samples,False) # index for union of existing Ai (r tensor)
        idx_Ai = idx_rest.repeat(len(acqf.best_f),1) # index for each Ai (n+1 x r tensor)
        for k in range(1,len(acqf.best_f)):
            if k==len(acqf.best_f)-1:
                idx_Ai[k,...] = ~idx_rest # A0 is the rest of the space
            else:
                idx_Ai[k,...] = Ai_index(acqf.best_c,k,weight_samples,False) & ~idx_rest # Ai = (c,infty)\ U Aj
                idx_rest = idx_rest | idx_Ai[k,...] # add Ai to the rest

        # compute int_Aj P(c > a)
        for j in range(len(acqf.best_f)):
            prob[...,j] = dist_l[idx_Ai[j,...],...].sum(dim=0).mul(1/acqf.raw_samples) # Monte Carlo est

    if len(acqf.con_upper_inds) > 0: # cond: c < a
        # dist_u: r samples of P(c < a) for each of the b batches (r x b tensor)
        dist_u = torch.ones(acqf.raw_samples, means[..., 0].shape[0])
        for i in acqf.con_upper_inds:
            # turn Gaussian samples into samples from p(a)
            ws = weight_samples[...,idx].repeat(means[..., 0].shape[0], 1).T # r x b


            # compute P(c_i < a_i) and multiply it to dist_u
            dist_u *= Phi((ws + acqf.con_upper[idx] - means[..., i]) / sigmas[..., i]) # r x b
            idx += 1 # increase s_w count

        # get indices for each Ai
        idx_rest = Ai_index(acqf.best_c,0,weight_samples,True) # index for union of existing Ai (r tensor)
        idx_Ai = idx_rest.repeat(len(acqf.best_f),1) # index for each Ai (n+1 x r tensor)

        for k in range(1,len(acqf.best_f)):
            if k==len(acqf.best_f)-1:
                idx_Ai[k,...] = ~idx_rest # A0 is the rest of the space
            else:
                idx_Ai[k,...] = Ai_index(acqf.best_c,k,weight_samples,True) & ~idx_rest # Ai = (c,infty)\ U Aj
                idx_rest = idx_rest | idx_Ai[k,...] # add Ai to the rest

        # compute int_Aj P(c < a)
        for j in range(len(acqf.best_f)):
            prob[...,j] = dist_u[idx_Ai[j,...],...].sum(dim=0).mul(1/acqf.raw_samples) # Monte Carlo est

    prob[prob != prob] = 0.0

    return prob



def Ai_index(
    best_c: Tensor,
    id: int,
    weight_samples: Tensor,
    upper: bool
) -> Tensor:
    r"""Indices of samples in (c1,infty) x ... x (cm,infty)
        Args:
            best_c:          n+1 x m Tensor of c values
            id:              index of order stat
            weight_samples:  r x m Tensor of samples
            upper:           bool if c<a or not
        Returns:
            r x m Tensor of indices of samples in (c1,infty) x ... x (cm,infty)
    """

    if upper:
        idx = weight_samples[...,0]>best_c[id,0]
        if weight_samples.shape[1]>1:
            for k in range(1,weight_samples.shape[1]):
                idx = idx & (weight_samples[...,k]>best_c[id,k])
    else:
        idx = weight_samples[...,0]<best_c[id,0]
        if weight_samples.shape[1]>1:
            for k in range(1,weight_samples.shape[1]):
                idx = idx & (weight_samples[...,k]<best_c[id,k])

    return idx


def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)
