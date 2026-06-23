import os
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from threadpoolctl import threadpool_limits


######################################################
####################  ESS   ##########################
######################################################

def _reflect_into_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, max_bounces: int = 3) -> np.ndarray:
    """Reflect x into [lb, ub] by bouncing off violated boundary faces.

    Each violated dimension is mirrored about the boundary (up to max_bounces
    times).  Falls back to clipping if still out of bounds after all bounces.
    Prevents the face-piling artefact that occurs when clipping is used after
    hyper-rectangle combination or go-beyond extrapolation.
    """
    x = x.copy()
    for _ in range(max_bounces):
        lo_viol = x < lb
        hi_viol = x > ub
        if not (lo_viol.any() or hi_viol.any()):
            return x
        x = np.where(lo_viol, 2.0 * lb - x, x)
        x = np.where(hi_viol, 2.0 * ub - x, x)
    return np.clip(x, lb, ub)


def _make_cached_objective(objective_fn, cache_size: int = 2000):
    """Wrap objective_fn with a fixed-size dict cache keyed on x.round(10).tobytes().

    Avoids re-evaluating duplicate points arising from bound reflections,
    RefSet churn, and stagnation resets.  Uses a FIFO eviction policy.

    Exposes _hits and _misses as single-element lists on the returned
    callable so _run_ess can print a cache hit-rate summary.
    """
    cache: dict = {}
    order: list = []
    _hits:   list = [0]
    _misses: list = [0]

    def _cached(x: np.ndarray) -> float:
        key = x.round(10).tobytes()
        if key in cache:
            _hits[0] += 1
            return cache[key]
        _misses[0] += 1
        val = objective_fn(x)
        if len(order) >= cache_size:
            oldest = order.pop(0)
            cache.pop(oldest, None)
        cache[key] = val
        order.append(key)
        return val

    _cached._hits   = _hits    # type: ignore[attr-defined]
    _cached._misses = _misses  # type: ignore[attr-defined]
    return _cached


def _run_local_search(
    x0: np.ndarray,
    objective_fn,
    residuals_fn,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    local_solver: str,
    n_params: int,
    rhoend: float = 1e-8,
    print_flag: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Run a local search from x0 using the configured solver.

    Returns (x_best, f_best, n_evals).  Dispatch order:
    1. dfo-ls  — derivative-free least-squares; needs residuals_fn; best for
                large models with sum-of-squares objectives.
    2. nelder-mead — always-available fallback; no residuals required.

    Parameters
    ----------
    rhoend : float
        Final trust-region radius for DFO-LS (1e-8 inner search, 1e-10 polish).
    """
    _solver = local_solver.lower()

    if _solver == "dfo-ls" and residuals_fn is not None:
        try:
            import dfols
            rhobeg = max(0.1, 2.0 * rhoend)
            soln = dfols.solve(
                residuals_fn,
                x0,
                bounds=(lb, ub),
                maxfun=budget,
                rhobeg=rhobeg,
                rhoend=rhoend,
                scaling_within_bounds=True,
                do_logging=False,
                print_progress=False,
            )
            f_soln = float(np.dot(soln.resid, soln.resid))
            if np.isfinite(f_soln):
                return soln.x.copy(), f_soln, int(soln.nf)
            # Non-finite result — fall through to Nelder-Mead.
        except Exception as e:
            if print_flag:
                print(f"  dfo-ls failed ({type(e).__name__}), falling back to Nelder-Mead")

    # Nelder-Mead fallback (no extra dependencies required).
    from scipy.optimize import minimize as _nm
    with threadpool_limits(limits=1, user_api=None):
        res = _nm(
            objective_fn,
            x0=x0,
            method="Nelder-Mead",
            bounds=list(zip(lb.tolist(), ub.tolist())),
            options={"maxfev": budget, "xatol": 1e-9, "fatol": 1e-9, "adaptive": True},
        )
    return res.x.copy(), float(res.fun), int(res.nfev)


def _run_ess(
    objective_fn,
    theta0_log: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    dim_refset: int = 10,
    local_n1: int = 1,
    local_n2: int = 2,
    balance: float = 0.5,
    max_eval: int = 100_000,
    max_walltime_s=None,
    vtr=None,
    n_stuck_global_limit: int = 0,
    use_finish_polish: bool = True,
    finish_polish_maxfev: int = 5000,
    on_new_best=None,
    inject_fn=None,
    inject_interval: int = 20,
    inject_warmup_iters: int = 50,
    theta0_include_prob: float = 1.0,
    print_iterations: bool = False,
    residuals_fn=None,
    local_solver: str = "nelder-mead",
    n_diverse_factor: int = 10,
    stagnation_perturb_sigma: float = 0.1,
    n_ls_children: int = 0,
    kick_limit: int = 3,
    trace_list: list | None = None,
) -> tuple[np.ndarray, float]:
    """
    Core Enhanced Scatter Search (ESS) algorithm — no file I/O.

    Implements the Egea & Banga (2009) ESS algorithm with the following
    improvements over the baseline:
    - DFO-LS / Nelder-Mead local search (dispatch via local_solver)
    - n_diverse scales with n_params for better high-dimensional coverage
    - RefSet seeded via half-quality / half-diversity split
    - Diversity distances normalised by (ub-lb) to avoid scale bias
    - Reflection at bounds (no face-piling from clipping)
    - Eval cache (FIFO, avoids redundant model calls)
    - Type-2 combination (midpoint + Gaussian) as 50/50 coin-flip with Type-1
    - Adaptive go-beyond: halves step on failure, allows one retry
    - Adaptive ls_threshold initialised at RefSet median (not best)
    - Mixed stagnation reset: half perturb-best, half fresh LHS
    - Hash-set de-dup and hash-map stagnation tracking (O(N) vs O(N^2))
    - Global-stagnation kick: basin hop when stuck, hard stop after kick_limit

    Parameters
    ----------
    objective_fn : callable
        Objective function f(x) -> float.  Lower is better.
    theta0_log : np.ndarray
        Initial parameter guess (optimized subspace, within bounds).
    lb, ub : np.ndarray
        Parameter bounds for the optimized subspace.
    residuals_fn : callable(x) -> np.ndarray | None
        Residual vector function used by DFO-LS.  If None, falls back to
        Nelder-Mead regardless of local_solver.
    local_solver : str
        Local search backend: "dfo-ls" (preferred) or "nelder-mead".
    n_diverse_factor : int
        n_diverse = max(factor * n_params, factor * dim_refset).
    stagnation_perturb_sigma : float
        Std-dev (as fraction of bound width) for the perturbation-of-best
        half of mixed stagnation resets.  The other half draws fresh LHS points.
    on_new_best : callable(cost, x) | None
        Called whenever a new global best is found (e.g. for checkpointing).
    inject_fn : callable() -> (cost, x) | None | None
        Called every ``inject_interval`` iterations.  If it returns a
        ``(cost, x)`` pair the point is considered for injection into the
        RefSet worst slot.  Pass ``None`` to disable.
    inject_interval : int
        How many iterations between ``inject_fn`` calls.

    Returns
    -------
    best_x : np.ndarray
        Best parameter vector found (optimized subspace).
    best_fval : float
        Corresponding objective value.
    """
    n_params = len(lb)

    # Normalised scale vector used for diversity distances and perturbations.
    _scale = np.where(ub > lb, ub - lb, 1.0)

    # Wrap objective with a small FIFO eval cache to skip identical re-evals.
    _cache_size  = max(2000, 20 * dim_refset)
    objective_fn = _make_cached_objective(objective_fn, _cache_size)

    # Auto-scale number of LS candidates per iteration.
    _n_ls_children = n_ls_children if n_ls_children > 0 else min(3, max(1, dim_refset // 4))
    # Global-stagnation kick — basin hop when stuck for _kick_threshold iters.
    _kick_threshold = max(15, 3 * dim_refset)
    # Start time for convergence trace.
    t0_run = time.time()

    # _best[0] = best cost seen, _best[1] = best x (optimized subspace).
    _best = [float('inf'), theta0_log.copy()]

    # --- Initialise RefSet ---
    # Draw n_diverse candidates via LHS; scale with n_params for adequate
    # coverage of the high-dimensional box.
    n_diverse      = max(n_diverse_factor * n_params, n_diverse_factor * dim_refset)
    include_theta0 = np.random.random() < theta0_include_prob
    if include_theta0:
        diverse_points = qmc.scale(qmc.LatinHypercube(d=n_params).random(n=n_diverse - 1), lb, ub)
        diverse_points = np.vstack([theta0_log.reshape(1, -1), diverse_points])
    else:
        diverse_points = qmc.scale(qmc.LatinHypercube(d=n_params).random(n=n_diverse), lb, ub)

    print(
        f"Starting ESS for pid {os.getpid()} "
        f"(n_params={n_params}, dim_refset={dim_refset}, max_eval={max_eval})")
    print(
        f"  Seeding RefSet from {n_diverse} diverse candidates"
        f" ({'with' if include_theta0 else 'without'} theta0)...")

    diverse_fx = np.array([objective_fn(pt) for pt in diverse_points])
    n_evals    = n_diverse

    # Seed RefSet: half quality (lowest cost) + half diversity (farthest
    # from quality in normalised space).  Mirrors the Phase 2 update logic
    # and prevents early basin collapse.
    n_quality_seed   = max(1, dim_refset // 2)
    n_diversity_seed = dim_refset - n_quality_seed
    order_seed       = np.argsort(diverse_fx)
    seed_idx         = list(order_seed[:n_quality_seed])
    seed_pts         = diverse_points[seed_idx]
    seed_remaining   = list(order_seed[n_quality_seed:])

    for _ in range(n_diversity_seed):
        if not seed_remaining:
            break
        _rem_pts    = diverse_points[seed_remaining]
        _d_to_seeds = cdist(_rem_pts / _scale, seed_pts / _scale)
        dists  = _d_to_seeds.min(axis=1)
        best_r = seed_remaining[int(np.argmax(dists))]
        seed_idx.append(best_r)
        seed_pts = np.vstack([seed_pts, diverse_points[best_r]])
        seed_remaining.remove(best_r)

    rs_x  = diverse_points[seed_idx].copy()
    rs_fx = diverse_fx[seed_idx].copy()

    if rs_fx[0] < _best[0]:
        _best[0] = rs_fx[0]
        _best[1] = rs_x[0].copy()
        if on_new_best is not None:
            on_new_best(_best[0], _best[1])

    if trace_list is not None:
        trace_list.append((n_evals, time.time() - t0_run, _best[0]))

    print(f"  RefSet seeded. Best = {rs_fx[0]:.6g}, worst = {rs_fx[-1]:.6g}")

    # n_stuck[i] = consecutive iterations member i was not updated.
    n_stuck      = np.zeros(dim_refset, dtype=int)
    stagnation_k = max(10, dim_refset)
    # Infeasible seed members are reset immediately on iteration 1.
    n_stuck[rs_fx >= 1e20] = stagnation_k

    n_kicks = 0  # Global-stagnation kick counter

    # Known local solutions (capped at 200) for the distance filter.
    local_solutions: list = []
    ls_maxdist:      list = []
    ls_wait_maxdist: list = []

    # Local search filter state.
    # Initialise threshold at RefSet median so Stage 2 fires earlier on
    # heavy models instead of waiting for a child that beats the global best.
    ls_stage1_done    = False
    ls_threshold      = rs_fx[dim_refset // 2]
    ls_wait_threshold = 0
    ls_wait_th_limit  = max(2, dim_refset // 4)
    ls_thfactor           = 0.1
    ls_wait_maxdist_limit = 5
    ls_maxdistfactor      = 0.1

    # Global stagnation tracking.
    n_iters_no_improve = 0
    _prev_best         = _best[0]

    n_iter          = 0
    last_local_iter = -local_n1

    t_start = time.time()

    # =====================================================================
    # Main ESS loop
    # =====================================================================
    with threadpool_limits(limits=1, user_api=None):
        while n_evals < max_eval:
            if max_walltime_s is not None and (time.time() - t_start) > max_walltime_s:
                print(f"  ESS wall-time limit reached at iter {n_iter} (pid {os.getpid()})")
                break

            if vtr is not None and _best[0] <= vtr:
                print(
                    f"  ESS VTR reached: {_best[0]:.6g} <= {vtr:.6g} "
                    f"(iter {n_iter}, pid {os.getpid()})")
                break

            if n_stuck_global_limit > 0 and n_iters_no_improve >= n_stuck_global_limit:
                print(
                    f"  ESS global stagnation: no improvement for "
                    f"{n_iters_no_improve} iters (pid {os.getpid()})")
                break

            n_iter += 1
            candidates_x      = []
            candidates_fx     = []
            candidates_parent = []  # RefSet index (i) of the parent for each candidate

            # -----------------------------------------------------------------
            # Phase 1: Combination + Go-Beyond for every ordered pair (i, j).
            #
            # Combination (hyper-rectangle, pyscat-style ≈ MATLAB Type 1):
            #   d     = (x[j] - x[i]) / 2
            #   alpha = +1 if i < j else -1        (direction bias)
            #   beta  = (|j-i| - 1) / (dim-2) ∈ [0,1]  (rank-distance scaling)
            #   c_lo  = x[i] - d*(1 + alpha*beta)
            #   c_hi  = x[i] + d*(1 - alpha*beta)
            #   x_comb ~ Uniform(min(c_lo,c_hi), max(c_lo,c_hi)) per dimension
            #
            # Out-of-bounds points are reflected (not clipped) to avoid face-piling.
            #
            # Go-beyond (stochastic + adaptive):
            #   Fires only if f_comb < rs_fx[i].
            #   Factor doubles after 2 consecutive wins; halves on failure with
            #   one retry allowed before the chain is terminated.
            # -----------------------------------------------------------------
            _denom_range = max(dim_refset - 2, 1)
            for i in range(dim_refset):
                for j in range(dim_refset):
                    if i == j:
                        continue

                    xi, xj = rs_x[i], rs_x[j]
                    fi     = rs_fx[i]

                    # Combination: Type-1 (hyper-rectangle) or Type-2 (midpoint +
                    # Gaussian perturbation). 50/50 coin-flip.
                    if np.random.random() < 0.5:
                        # Type-1: hyper-rectangle.
                        d     = (xj - xi) / 2.0
                        alpha = 1 if i < j else -1
                        beta  = (abs(j - i) - 1) / _denom_range
                        c_lo  = xi - d * (1.0 + alpha * beta)
                        c_hi  = xi + d * (1.0 - alpha * beta)
                        x_comb = np.random.uniform(
                            np.minimum(c_lo, c_hi),
                            np.maximum(c_lo, c_hi),
                        )
                    else:
                        # Type-2: midpoint + Gaussian (increases child diversity).
                        x_comb = 0.5 * (xi + xj) + np.random.normal(0.0, 0.25 * _scale)
                    x_comb = _reflect_into_bounds(x_comb, lb, ub)
                    f_comb = objective_fn(x_comb)
                    n_evals += 1
                    candidates_x.append(x_comb)
                    candidates_fx.append(f_comb)
                    candidates_parent.append(i)

                    # Go-beyond: only when child beats parent i.
                    if f_comb >= fi:
                        if n_evals >= max_eval:
                            break
                        continue

                    go_beyond_factor = 1.0
                    gb_improvement   = 0
                    gb_fail_count    = 0   # consecutive failures; halve factor, allow 1 retry
                    x_prev, x_cur, f_cur = xi, x_comb, f_comb
                    for _ in range(10):  # cap chain length
                        step    = (x_cur - x_prev) * go_beyond_factor
                        x_upper = _reflect_into_bounds(x_cur + step, lb, ub)
                        x_gb    = np.random.uniform(
                            np.minimum(x_cur, x_upper),
                            np.maximum(x_cur, x_upper),
                        )
                        x_gb   = _reflect_into_bounds(x_gb, lb, ub)
                        f_gb   = objective_fn(x_gb)
                        n_evals += 1
                        candidates_x.append(x_gb)
                        candidates_fx.append(f_gb)
                        candidates_parent.append(i)
                        if f_gb < f_cur:
                            x_prev, x_cur, f_cur = x_cur, x_gb, f_gb
                            gb_improvement += 1
                            gb_fail_count   = 0
                            if gb_improvement == 2:
                                go_beyond_factor *= 2.0
                                gb_improvement    = 0
                        else:
                            go_beyond_factor *= 0.5
                            gb_fail_count    += 1
                            if gb_fail_count >= 2:
                                break
                        if n_evals >= max_eval:
                            break

                    if n_evals >= max_eval:
                        break
                if n_evals >= max_eval:
                    break

            # -----------------------------------------------------------------
            # Phase 2: RefSet update — quality-diversity split.
            # First half: best dim_refset//2 candidates by cost (quality slots).
            # Second half: greedily chosen for maximum normalised distance from
            # quality slots (diversity slots).
            # Deduplication uses a hash-set (O(N)) rather than np.unique (O(N log N)).
            # Stagnation tracking uses a hash-map (O(dim_refset)) rather than
            # nested all-close loops (O(dim_refset^2)).
            # -----------------------------------------------------------------
            all_x  = np.vstack([rs_x] + [c.reshape(1, -1) for c in candidates_x])
            all_fx = np.concatenate([rs_fx, candidates_fx])

            # Hash-set deduplication.
            _seen: set = set()
            _keep: list = []
            for _i, _row in enumerate(all_x):
                _key = _row.round(12).tobytes()
                if _key not in _seen:
                    _seen.add(_key)
                    _keep.append(_i)
            all_x  = all_x[_keep]
            all_fx = all_fx[_keep]

            order       = np.argsort(all_fx)
            all_x       = all_x[order]
            all_fx      = all_fx[order]
            n_quality   = max(1, dim_refset // 2)
            n_diversity = dim_refset - n_quality

            # Quality slots: best n_quality points.
            selected_idx = list(range(n_quality))

            # Diversity slots: farthest-point selection with normalised distance.
            if n_diversity > 0 and len(all_x) > n_quality:
                remaining    = list(range(n_quality, len(all_x)))
                selected_pts = all_x[selected_idx]
                for _ in range(n_diversity):
                    if not remaining:
                        break
                    _rem_pts = all_x[remaining]
                    _d_mat   = cdist(_rem_pts / _scale, selected_pts / _scale)
                    dists = _d_mat.min(axis=1)
                    best_r = remaining[int(np.argmax(dists))]
                    selected_idx.append(best_r)
                    selected_pts = np.vstack([selected_pts, all_x[best_r]])
                    remaining.remove(best_r)

            new_rs_x  = all_x[selected_idx].copy()
            new_rs_fx = all_fx[selected_idx].copy()

            # Stagnation tracking via hash-map: O(dim_refset).
            _old_key_to_stuck = {
                rs_x[orig].round(12).tobytes(): int(n_stuck[orig])
                for orig in range(dim_refset)
            }
            new_n_stuck = np.zeros(dim_refset, dtype=int)
            for k in range(dim_refset):
                prev = _old_key_to_stuck.get(new_rs_x[k].round(12).tobytes())
                if prev is not None:
                    new_n_stuck[k] = prev + 1
            n_stuck = new_n_stuck

            rs_x  = new_rs_x
            rs_fx = new_rs_fx

            # Update global best if RefSet improved.
            if rs_fx[0] < _best[0]:
                _best[0] = rs_fx[0]
                _best[1] = rs_x[0].copy()
                if on_new_best is not None:
                    on_new_best(_best[0], _best[1])
                if trace_list is not None:
                    trace_list.append((n_evals, time.time() - t0_run, _best[0]))
                if print_iterations:
                    print(
                        f"  ESS iter {n_iter}: new best = {_best[0]:.6g} "
                        f"(evals: {n_evals}, pid {os.getpid()})")

            # -----------------------------------------------------------------
            # Phase 3: Local search dispatched via _run_local_search.
            # Cap per-LS budget vs. remaining eval budget.
            # Run LS on up to _n_ls_children distance-eligible candidates.
            #
            # Stage 1 (first call): single best-cost child, no filters.
            # Stage 2+: distance filter → composite merit+distance rank; top-k
            #           candidates are each given budget // k evals.
            # Fallback: best RefSet member if no child is eligible.
            # -----------------------------------------------------------------
            if n_iter >= local_n1 and (n_iter - last_local_iter) >= local_n2:
                last_local_iter = n_iter


                local_budget = max(30 * n_params, finish_polish_maxfev // 20)
                local_budget = min(local_budget, max(50, (max_eval - n_evals) // 5))

                ls_queue: list = []

                if not ls_stage1_done:
                    if candidates_fx:
                        best_c = int(np.argmin(candidates_fx))
                        ls_queue = [(candidates_x[best_c], candidates_parent[best_c])]
                        ls_stage1_done = True
                        ls_threshold   = min(float(candidates_fx[best_c]), _best[0])
                else:
                    _candidates_arr = np.array(candidates_x)

                    _all_dists: np.ndarray | None = None
                    if local_solutions:
                        _loc_arr    = np.array(local_solutions)
                        _diff       = (_candidates_arr[:, None, :] - _loc_arr[None, :, :]) / _scale
                        _all_dists  = np.linalg.norm(_diff, axis=2)
                        _md_arr     = np.array(ls_maxdist)
                        passes_dist = np.all(_all_dists > _md_arr[None, :], axis=1)
                        dist_eligible = list(np.where(passes_dist)[0])
                    else:
                        dist_eligible = list(range(len(candidates_x)))

                    if dist_eligible:
                        elig_fx    = np.array([candidates_fx[k] for k in dist_eligible])
                        merit_rank = np.argsort(np.argsort(elig_fx)).astype(float)
                        if local_solutions and _all_dists is not None:
                            elig_dists = np.min(
                                _all_dists[np.array(dist_eligible)], axis=1)
                            dist_rank = np.argsort(np.argsort(-elig_dists)).astype(float)
                        else:
                            dist_rank = np.zeros(len(dist_eligible))

                        scores = (1.0 - balance) * merit_rank + balance * dist_rank
                        n_to_pick   = min(_n_ls_children, len(dist_eligible))
                        chosen_rels = np.argsort(scores)[:n_to_pick]

                        first_ci = dist_eligible[int(chosen_rels[0])]
                        first_fx = float(candidates_fx[first_ci])
                        if first_fx < ls_threshold:
                            ls_wait_threshold = 0
                            ls_threshold      = min(ls_threshold, first_fx)
                        else:
                            ls_wait_threshold += 1
                            if ls_wait_threshold >= ls_wait_th_limit:
                                ls_threshold += ls_thfactor * (1.0 + abs(ls_threshold))
                                ls_wait_threshold = 0

                        ls_queue = [
                            (candidates_x[dist_eligible[r]], candidates_parent[dist_eligible[r]])
                            for r in chosen_rels
                        ]

                        if local_solutions and _all_dists is not None:
                            for m_idx in range(len(local_solutions)):
                                if _all_dists[first_ci, m_idx] <= ls_maxdist[m_idx]:
                                    ls_wait_maxdist[m_idx] += 1
                                    if ls_wait_maxdist[m_idx] >= ls_wait_maxdist_limit:
                                        ls_maxdist[m_idx] *= (1.0 - ls_maxdistfactor)
                                        ls_wait_maxdist[m_idx] = 0
                                    break
                                else:
                                    ls_wait_maxdist[m_idx] = 0

                if not ls_queue:
                    ls_queue = [(rs_x[0], 0)]

                ls_budget_each = max(50, local_budget // max(1, len(ls_queue)))
                for x0_local, parent_for_ls in ls_queue:
                    _rem = max_eval - n_evals
                    if _rem < 50:
                        break
                    per_budget = min(ls_budget_each, max(50, _rem // 5))
                    x_ls, f_ls, ls_nfev = _run_local_search(
                        x0_local, objective_fn, residuals_fn,
                        lb, ub, per_budget, local_solver, n_params,
                        rhoend=1e-8, print_flag=print_iterations,
                    )
                    n_evals += ls_nfev

                    if np.isfinite(f_ls):
                        is_new = not any(
                            np.linalg.norm((x_ls - x_loc) / _scale) < 1e-2
                            for x_loc in local_solutions
                        )
                        if is_new:
                            local_solutions.append(x_ls)
                            ls_maxdist.append(0.25)
                            ls_wait_maxdist.append(0)
                            if len(local_solutions) > 200:
                                local_solutions.pop(0)
                                ls_maxdist.pop(0)
                                ls_wait_maxdist.pop(0)

                        ls_threshold = min(ls_threshold, f_ls)

                        if parent_for_ls is not None and f_ls < rs_fx[parent_for_ls]:
                            rs_x[parent_for_ls]    = x_ls
                            rs_fx[parent_for_ls]   = f_ls
                            n_stuck[parent_for_ls] = 0
                        elif f_ls < rs_fx[-1]:
                            rs_x[-1]    = x_ls
                            rs_fx[-1]   = f_ls
                            n_stuck[-1] = 0
                        order   = np.argsort(rs_fx)
                        rs_x    = rs_x[order]
                        rs_fx   = rs_fx[order]
                        n_stuck = n_stuck[order]

                        if f_ls < _best[0]:
                            _best[0] = f_ls
                            _best[1] = x_ls
                            if on_new_best is not None:
                                on_new_best(_best[0], _best[1])
                            if trace_list is not None:
                                trace_list.append((n_evals, time.time() - t0_run, _best[0]))
                            if print_iterations:
                                print(
                                    f"  ESS iter {n_iter}: local search improved to "
                                    f"{_best[0]:.6g} (evals: {n_evals}, pid {os.getpid()})")

            # -----------------------------------------------------------------
            # Phase 4: Mixed stagnation reset.
            # Half of stuck members are replaced by perturbations of the current
            # best (intensification); the other half get fresh LHS points
            # (diversification).
            # -----------------------------------------------------------------
            for k in range(dim_refset):
                if n_stuck[k] >= stagnation_k:
                    if np.random.random() < 0.5:
                        # Intensification: perturb the current best.
                        sigma     = stagnation_perturb_sigma * _scale
                        rs_x[k]   = _reflect_into_bounds(
                            rs_x[0] + np.random.normal(0.0, sigma), lb, ub)
                    else:
                        # Diversification: fresh LHS draw.
                        rs_x[k] = np.clip(
                            qmc.scale(qmc.LatinHypercube(d=n_params).random(n=1), lb, ub)[0],
                            lb, ub)
                    rs_fx[k]   = objective_fn(rs_x[k])
                    n_evals   += 1
                    n_stuck[k] = 0

            # Re-sort so rs_fx[-1] is always the worst (needed for injection gates).
            order   = np.argsort(rs_fx)
            rs_x    = rs_x[order]
            rs_fx   = rs_fx[order]
            n_stuck = n_stuck[order]

            # -----------------------------------------------------------------
            # Phase 5: External injection — pick up improvements via inject_fn.
            # Inject into worst RefSet slot only; never promote to
            # _best directly.  This prevents the monoculture / checkpoint-feedback
            # loop where every worker converges to the same basin.
            # Warmup guard — skip injection for the first
            # inject_warmup_iters iterations so each worker explores independently.
            # -----------------------------------------------------------------
            if (inject_fn is not None
                    and n_iter % inject_interval == 0
                    and n_iter >= inject_warmup_iters):
                injected = inject_fn()
                if injected is not None:
                    inj_cost, inj_x = injected
                    if inj_cost < rs_fx[-1]:
                        rs_x[-1]    = inj_x
                        rs_fx[-1]   = inj_cost
                        n_stuck[-1] = 0
                        order = np.argsort(rs_fx)
                        rs_x  = rs_x[order]
                        rs_fx = rs_fx[order]
                        n_stuck = n_stuck[order]
                        if print_iterations:
                            print(
                                f"  ESS inject: {inj_cost:.6g} entered RefSet "
                                f"(iter {n_iter}, pid {os.getpid()})")

            # Update global stagnation counter.
            if _best[0] < _prev_best:
                n_iters_no_improve = 0
                _prev_best         = _best[0]
            else:
                n_iters_no_improve += 1

            # Global-stagnation kick — basin hop when stuck for _kick_threshold iters.
            # Replace bottom half of RefSet with LHS-around-best, reset counter.
            # Hard-stop after kick_limit kicks.
            if n_iters_no_improve >= _kick_threshold:
                n_kicks += 1
                if kick_limit > 0 and n_kicks > kick_limit:
                    print(
                        f"  ESS kick limit ({kick_limit}) reached — stopping "
                        f"(pid {os.getpid()})")
                    break
                _n_kick_slots = dim_refset // 2
                _sigma_kick   = 0.5 * _scale
                for _k in range(dim_refset - _n_kick_slots, dim_refset):
                    rs_x[_k]    = _reflect_into_bounds(
                        rs_x[0] + np.random.normal(0.0, _sigma_kick), lb, ub)
                    rs_fx[_k]   = objective_fn(rs_x[_k])
                    n_evals    += 1
                    n_stuck[_k] = 0
                order   = np.argsort(rs_fx)
                rs_x    = rs_x[order]
                rs_fx   = rs_fx[order]
                n_stuck = n_stuck[order]
                n_iters_no_improve = 0
                if print_iterations:
                    print(
                        f"  ESS kick #{n_kicks}: replaced bottom {_n_kick_slots} slots "
                        f"(evals: {n_evals}, pid {os.getpid()})")

            # Separate hard-stop on n_stuck_global_limit (0 = disabled).
            if n_stuck_global_limit > 0 and n_iters_no_improve >= n_stuck_global_limit:
                print(
                    f"  ESS global stagnation: no improvement for "
                    f"{n_iters_no_improve} iters (pid {os.getpid()})")
                break

    print(
        f"Done with ESS for pid {os.getpid()} — best f = {_best[0]:.6g} "
        f"(total evals: {n_evals}, iters: {n_iter})")

    best_fval = _best[0]
    best_x    = _best[1]

    # Final finish polish on overall best — tighter tolerance than inner search.
    if use_finish_polish:
        _polish_budget = min(finish_polish_maxfev, max(500 * n_params, max_eval - n_evals))
        if print_iterations:
            print(f"  Running final finish polish (pid {os.getpid()})")

        x_pol, f_pol, _ = _run_local_search(
            best_x, objective_fn, residuals_fn,
            lb, ub, _polish_budget, local_solver, n_params,
            rhoend=1e-10, print_flag=print_iterations,
        )
        if f_pol < best_fval:
            if print_iterations:
                print(f"  Finish polish improved: {best_fval:.6g} -> {f_pol:.6g}")
            best_fval = f_pol
            best_x    = x_pol
            if on_new_best is not None:
                on_new_best(best_fval, best_x)
            if trace_list is not None:
                trace_list.append((n_evals, time.time() - t0_run, best_fval))

    # Print eval-cache hit-rate when telemetry is available.
    if print_iterations and hasattr(objective_fn, '_hits'):
        _h = objective_fn._hits[0]       # type: ignore[attr-defined]
        _m = objective_fn._misses[0]     # type: ignore[attr-defined]
        _tot = _h + _m
        if _tot > 0:
            print(
                f"  Eval cache: {_h}/{_tot} hits ({100*_h/_tot:.1f}%) "
                f"(pid {os.getpid()})")

    return best_x, best_fval
