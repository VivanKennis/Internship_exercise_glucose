"""
Legacy parameter estimation optimizers — archived when ESS became the default.

These functions require the helpers from the parent package (common.parameter_estimation,
common.cost_functions, common.utils) and are kept here for reference only.
They are NOT imported by the active codebase.
"""

# ============================================================================
# optimize_cma  (formerly optimize_pypesto)
# ============================================================================

def optimize_pypesto(input_args):
    """
    Run multi-start CMA-ES optimisation using the cma package directly,
    followed by a Nelder-Mead local polish after each CMA-ES run.

    Calls cma.CMAEvolutionStrategy in a plain loop over n_starts restarts.
    No pyPESTO scheduler, engine, or deep-copy machinery is involved, so SUND
    simulation objects (which forbid deep copying) are never copied.

    Each restart begins near theta0_log (±5 % in log space) and is clipped to
    the parameter bounds.  The best result across all restarts is saved in the
    same JSON format as optimize() so load_best_parameters() needs no changes.
    """
    import cma
    import functools
    from scipy.optimize import minimize as scipy_minimize

    def cma_callback(es):
        if es.result.fbest < 1e20 and es.result.fbest < best_fval:
            try:
                p_full = reconstruct_parameter_vector(es.result.xbest, theta0_constants_log, constant_indices)
                save_result(es.result.fbest, p_full, temp_filename)
            except Exception as e:
                print(f"Error saving CMA-ES checkpoint: {e}")

    
    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())
    
    # unpack input arguments
    data                     = input_args['data']
    fixed_parameters         = input_args['fixed_params']
    model_name               = input_args['model_name']
    strict_bounds_parameters = input_args['strict_params']
    n_starts                 = input_args['n_starts']
    max_iter                 = input_args['max_iter']
    simulate_steady_state    = input_args['simulate_steady_state']
    sigma0                   = input_args.get('sigma0', 0.3)
    use_local_polish         = input_args.get('use_local_polish', True)
    local_polish_maxfev      = input_args.get('local_polish_maxfev', 2000)
    local_polish_balance     = input_args.get('local_polish_balance', 0.5)
    use_finish_polish        = input_args.get('use_finish_polish', True)
    finish_polish_maxfev     = input_args.get('finish_polish_maxfev', 5000)
    print_iterations         = input_args.get('print_iter', True)
    
    # temporary file to store the best parameters found so far
    temp_filename = f"./results/{model_name}/{model_name}-temp-{os.getpid()}.json"

    # setup simulations
    sims = setup_simulations(model_name, data)
    sim = next(iter(sims.values()))

    theta0 = load_best_parameters(
        f"./results/{model_name}", cost_key='f')

    optimize_indices, constant_indices = get_parameter_indices_to_optimize(
        sim.parameter_names, fixed_parameters)

    # divide start guess in a part to optimize and a part to remain constant
    theta0_log = np.log(np.array(theta0)[optimize_indices])
    theta0_constants_log = np.log(np.array(theta0)[constant_indices])

    # set up parameter bounds
    bounds = get_parameter_bounds(
        sim.parameter_names, theta0_log, optimize_indices, strict_bounds_parameters)
    lb = np.array(bounds.lb)
    ub = np.array(bounds.ub)
    
    # Bind fixed-parameter context so the callable only sees the optimized sub-vector
    objective_fn = functools.partial(
        f_cost_log_with_fixed_parameters,
        sims=sims,
        data=data,
        simulate_steady_state=simulate_steady_state,
        theta0_constant=theta0_constants_log,
        constant_indices=constant_indices
    )

    n_params = len(theta0_log)

    # Scale polish budgets with problem size so they remain sufficient for
    # large models (each Nelder-Mead iteration costs ~n+1 evaluations).
    local_polish_maxfev  = max(local_polish_maxfev,  200 * n_params)
    finish_polish_maxfev = max(finish_polish_maxfev, 500 * n_params)

    # Auto-scale CMA-ES maxiter: target ~1000*n/popsize total evaluations.
    # MAXITER from config acts as a minimum so it can still be raised manually.
    popsize = max(50, int(10 * np.sqrt(n_params)))
    auto_maxiter = max(max_iter, (1000 * n_params) // popsize)

    cma_options = {
        "bounds":   [lb.tolist(), ub.tolist()],
        "maxiter":  auto_maxiter,
        "verbose":  0 if print_iterations else -1,
        "tolfun":   1e-9,
        "tolx":     1e-9,
        "tolstagnation": max(500, 100 + 100 * n_params // 50),
        "popsize":       popsize,
    }

    # Sanity-check the initial guess before launching CMA-ES
    test_cost = objective_fn(theta0_log)
    if test_cost >= 1e20:
        print(f"  WARNING: test evaluation at theta0 returned {test_cost:.3e}. "
              f"Check parameter assembly and SUND model. Aborting CMA-ES.")
        return
    print(f"  Test evaluation at theta0: f = {test_cost:.6g}")

    best_x    = theta0_log.copy()
    best_fval = test_cost
    
    print(f"Starting CMA-ES with {n_starts} starts for pid {os.getpid()}")
    for i in range(n_starts):
        if np.random.random_sample() < input_args['perturbation_ratio']:
            x0 = theta0_log * np.random.uniform(0.95, 1.05, n_params)
        else:
            x0 = theta0_log.copy()
        
        # make sure that the start guess is within bounds
        x0 = np.clip(x0, lb, ub)

        print(f"  CMA-ES start {i + 1}/{n_starts} (pid {os.getpid()})")
        res = cma.CMAEvolutionStrategy(x0, sigma0, inopts=cma_options).optimize(objective_fn, callback=cma_callback).result

        if res.fbest >= 1e20:
            continue

        candidate_fval = res.fbest
        candidate_x    = np.array(res.xbest)

        # --- Local polish -----------------------------------
        if use_local_polish:
            step = sigma0 * (0.1 + local_polish_balance * 0.9)
            initial_simplex = np.zeros((n_params + 1, n_params))
            initial_simplex[0] = candidate_x
            for j in range(n_params):
                row = candidate_x.copy()
                row[j] = np.clip(row[j] + step, lb[j], ub[j])
                initial_simplex[j + 1] = row

            local_res = scipy_minimize(
                objective_fn,
                x0=candidate_x,
                method='Nelder-Mead',
                bounds=list(zip(lb.tolist(), ub.tolist())),
                options={
                    'maxfev': local_polish_maxfev,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True,
                    'initial_simplex': initial_simplex,
                },
            )
            if local_res.fun < candidate_fval:
                candidate_fval = local_res.fun
                candidate_x    = local_res.x
        # ----------------------------------------------------

        if candidate_fval < best_fval:
            best_fval = candidate_fval
            best_x    = candidate_x

            p_full = reconstruct_parameter_vector(best_x, theta0_constants_log, constant_indices)
            save_result(best_fval, p_full, temp_filename)
        print(f"  start {i + 1} best: {candidate_fval:.6g} (overall best: {best_fval:.6g})")

    print(f"Done with CMA-ES for pid {os.getpid()} — best f = {best_fval}")

    # Final finish polish on overall best.
    if use_finish_polish:
        finish_res = scipy_minimize(
            objective_fn,
            x0=best_x,
            method='Nelder-Mead',
            bounds=list(zip(lb.tolist(), ub.tolist())),
            options={'maxfev': finish_polish_maxfev, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
        )
        if finish_res.fun < best_fval:
            best_fval = finish_res.fun
            best_x    = finish_res.x

    # Reconstruct full parameter vector and save in the standard JSON format
    p_full = reconstruct_parameter_vector(best_x, theta0_constants_log, constant_indices)
    file_name = (
        f"./results/{model_name}/{model_name} ({best_fval}) - "
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    save_result(best_fval, p_full, file_name)

    # Remove the temporary file created by callback functions
    if os.path.exists(temp_filename):
        os.remove(temp_filename)


# ============================================================================
# optimize  (Differential Evolution / Dual Annealing)
# ============================================================================

def optimize(input_args: dict) -> None:
    def callback(x, convergence) -> None:
        nonlocal best_cost
        cost = f_cost_log_with_fixed_parameters(
            x, sims, data, simulate_steady_state, theta0_constants_log, constant_indices)
        
        if cost < best_cost:
            best_cost = cost
            p_full = reconstruct_parameter_vector(
                x, theta0_constants_log, constant_indices)
            save_result(cost, p_full, temp_filename)

    def callback_dual_annealing(x, cost: float, context) -> None:
        nonlocal n_iter, best_cost
        if print_iterations:
            print(f"dual_annealing step {n_iter}: f(x) = {cost} - pid= {os.getpid()}")
        n_iter += 1
        
        if cost < best_cost:
            best_cost = cost
            p_full = reconstruct_parameter_vector(
                x, theta0_constants_log, constant_indices)
            save_result(cost, p_full, temp_filename)

    import sys
    from scipy.optimize import differential_evolution, dual_annealing
    from common.utils import silent_errors

    np.random.seed(int(time.time()) + os.getpid())

    data                     = input_args['data']
    fixed_parameters         = input_args['fixed_params']
    model_name               = input_args['model_name']
    simulate_steady_state    = input_args['simulate_steady_state']
    strict_bounds_parameters = input_args['strict_params']
    print_iterations         = input_args['print_iter']
    
    sims = setup_simulations(model_name, data)
    sim = next(iter(sims.values()))
    
    temp_filename = f"./results/{model_name}/{model_name}-temp-{os.getpid()}.json"
    
    theta0 = load_best_parameters(f"./results/{model_name}", cost_key='f')
    
    optimize_indices, constant_indices = get_parameter_indices_to_optimize(
        sim.parameter_names, fixed_parameters)

    theta0_log = np.log(np.array(theta0)[optimize_indices])
    theta0_constants_log = np.log(np.array(theta0)[constant_indices])

    best_cost = f_cost_log_with_fixed_parameters(
        theta0_log, sims, data, simulate_steady_state, theta0_constants_log, constant_indices)
    
    bounds = get_parameter_bounds(
        sim.parameter_names, theta0_log, optimize_indices, strict_bounds_parameters)

    if np.random.random_sample() < input_args['perturbation_ratio']:
        theta0_log = theta0_log * np.random.uniform(0.95, 1.05, len(theta0_log))

    theta0_log = np.clip(theta0_log, bounds.lb, bounds.ub)

    args = (sims, data, simulate_steady_state, theta0_constants_log, constant_indices)

    do_differential_evolution = bool(np.random.random_sample() <= input_args['de_ratio'])

    if do_differential_evolution:
        print(f'Starting differential_evolution for pid {os.getpid()}')
        init = create_init_pop(theta0_log, bounds, input_args['pop_size'])
        with silent_errors(sys.stderr, os.devnull):
            res = differential_evolution(
                func=f_cost_log_with_fixed_parameters, args=args, bounds=bounds,
                x0=theta0_log, disp=print_iterations, maxiter=input_args['max_iter'], 
                callback=callback, init=init)
    else:
        bounds = bounds_to_tuples(bounds.lb, bounds.ub, len(bounds.lb))
        print(f'Starting dual_annealing for pid {os.getpid()}')
        n_iter = 1
        with silent_errors(sys.stderr, os.devnull):
            res = dual_annealing(
                func=f_cost_log_with_fixed_parameters, args=args, bounds=bounds,
                x0=theta0_log, maxiter=input_args['max_iter'], callback=callback_dual_annealing)
        print(f'Done with dual_annealing for pid {os.getpid()}')

    file_name = f"./results/{model_name}/{model_name} ({res['fun']}) - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p_full = reconstruct_parameter_vector(res['x'], theta0_constants_log, constant_indices)
    save_result(res['fun'], p_full, file_name)
    cleanup_temp_file(file_path=temp_filename)
