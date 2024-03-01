# Plotting functions
from Packages import *
from BO_functions import *

def plot_1d(ax1,ax2,fn, input_bounds = torch.tensor([[0.0], [1.0]])):
    """ PLotting 1d objective and constraint """
    # fn: function to plot

    test_X = torch.linspace(input_bounds[0].item(), input_bounds[1].item(), 101).unsqueeze(-1)
    ax1.plot(test_X, observe(test_X,fn)[:,0])

    ax2.plot(test_X, observe(test_X,fn)[:,1])
    ax2.plot(input_bounds[:,0], [0, 0], color="black")



def plot_2d(fn, input_bounds):
    """ PLotting 2d objective and constraint """
    # fn: function to plot

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    x = np.arange(input_bounds[0,0], input_bounds[0,1], 0.1)
    y = np.arange(input_bounds[1,0], input_bounds[1,1], 0.1)

    X, Y = np.meshgrid(x, y)
    Z = X*Y
    W = X*Y
    for i in range(0,len(y)):
        temp = torch.rand(len(x),2)
        temp[...,0] = torch.tensor(x)
        temp[...,1] = y[i]
        Z[i,...] = fn(temp)[...,0]
        W[i,...] = fn(temp)[...,1]

    ax[1].contourf(X, Y, W)
    ax[0].contourf(X, Y, Z)

    fig.tight_layout()



def plot_gp_posterior_and_acq_func(fn,gp, acq_func, train_X, train_y, gp_ax, ctrn_ax, acq_ax, idx=None):
    """ Plot acquired points and acquisition function for 1d"""
    # fn: function to plot
    # gp: the posterior GP model
    # acq_func: a module that computes the acquisition function
    # train_X: [N x 1] matrix of inputs
    # train_y: [N x 2] matrix of objective and constraint functions observations at input locations
    # gp_ax, ctrn_ax and acq_ax: figure axes to plot results on
    # idx: iteration index (for title)

    input_bounds = torch.tensor([[0.0], [1.0]])

    # Define points to evaluate our surrogate model and acq function at
    test_X = torch.linspace(input_bounds[0, 0].item(), input_bounds[1, 0].item(), 101).unsqueeze(-1)
    # A [101 x 1] matrix, spanning the compact domain that we are considering

    # Compute GP posterior @ test_X
    with torch.no_grad():
        gp.eval()
        test_f_posterior = gp.posterior(test_X)
        mean = test_f_posterior.mean[...,0].squeeze(-1)
        stdv = test_f_posterior.stddev[...,0].squeeze(-1)
        mean_ctrn = test_f_posterior.mean[...,1].squeeze(-1)
        stdv_ctrn = test_f_posterior.stddev[...,1].squeeze(-1)

    # Compute acquisition function @ test_X
    test_X_expanded = test_X[..., None, :]
    if isinstance(acq_func, botorch.acquisition.knowledge_gradient.qKnowledgeGradient):
        acq_test_X = acq_func.evaluate(
            test_X_expanded, bounds=input_bounds, num_restarts=2, raw_samples=128,
        ).detach()
    else:
        with torch.no_grad():
            acq_test_X = acq_func(test_X_expanded)

    # Plot gp
    gp_ax.plot(test_X.numpy(), fn(test_X)[..., 0].numpy(), "k--", zorder=-2, label="True Objective")
    mean_line, = gp_ax.plot(test_X.numpy(), mean.numpy(), color="blue", zorder=0, label="Posterior Mean")
    gp_ax.fill_between(
        test_X[..., 0].numpy(), (mean - 2 * stdv).numpy(), (mean + 2 * stdv).numpy(),
        color=mean_line.get_color(), zorder=-1, alpha=0.33
    )
    gp_ax.scatter(
        train_X[..., 0].numpy(), train_y[..., 0].numpy(),
        marker="o", edgecolors="k", color="white", label="Train Data"
    )
    gp_ax.set(xlabel="x", ylabel="f(x)", title=f"Iteration {idx}" if idx is not None else None)
    if idx == 0:
        gp_ax.legend(loc="best")

    # Plot constrains
    ctrn_ax.plot(test_X.numpy(), fn(test_X)[..., 1].numpy(), "k--", zorder=-2, label="Constraint function")
    mean_ctrn_line, = ctrn_ax.plot(test_X.numpy(), mean_ctrn.numpy(), color="blue", zorder=0, label="Posterior Mean")
    ctrn_ax.fill_between(
        test_X[..., 0].numpy(), (mean_ctrn - 2 * stdv_ctrn).numpy(), (mean_ctrn + 2 * stdv_ctrn).numpy(),
        color=mean_ctrn_line.get_color(), zorder=-1, alpha=0.33
    )
    ctrn_ax.scatter(
        train_X[..., 0].numpy(), train_y[..., 1].numpy(),
        marker="o", edgecolors="k", color="white", label="Train Data"
    )
    ctrn_bound, = ctrn_ax.plot([input_bounds[0, 0].item(), input_bounds[1, 0].item()], [0, 0], color="black")
    ctrn_ax.set(xlabel="x", ylabel="c(x)", title=f"Iteration {idx}" if idx is not None else None)
    if idx == 0:
        ctrn_ax.legend(loc="best")

    # Plot acq func
    idx_max = acq_test_X.argmax(dim=-1)
    acq_ax.plot(test_X.numpy(), acq_test_X.numpy(), color="green", zorder=0, label="Acq. Func")
    acq_ax.scatter(
        test_X[idx_max].numpy(), acq_test_X[idx_max].numpy(),
        marker="s", edgecolors="k", color="white", label="Maximizer"
    )
    acq_ax.set(xlabel="x", ylabel="a(x | f, D)", title=f"Iteration {idx}" if idx is not None else None)
    if idx == 0:
        acq_ax.legend(loc="best")



def plot_reward(reward_ci,con_reward_ci,feas_count,ax_r,ax_cr,ax_feas,name,marker=","):
    """ Plot reward curve and feasible proportion """
    # reward_ci: T x m+1 matrix of reward mean and confidence interval
    # con_reward_ci: T x m+1 matrix of constrained reward mean and confidence interval
    # feas_count: T vector of feasible count mean and confidence interval
    # ax_reward, ax_feas: plots for reward, constrained reward and feasible count

    T = len(feas_count)

    ax_r.plot(reward_ci[:,1], label = name, marker = marker)
    ax_r.fill_between(range(T),reward_ci[:,0], reward_ci[:,2], alpha=.5, linewidth=0)
    ax_cr.plot(con_reward_ci[:,1], marker = marker)
    ax_cr.fill_between(range(T),con_reward_ci[:,0], con_reward_ci[:,2], alpha=.5, linewidth=0)
    ax_feas.plot(feas_count[:,1]/torch.range(1,T), marker = marker)
    ax_feas.fill_between(range(T),feas_count[:,0]/torch.range(1,T), feas_count[:,2]/torch.range(1,T), alpha=.5, linewidth=0)

    return ax_r, ax_cr, ax_feas

