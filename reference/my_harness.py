"""Remove unused final ensemble transitions without changing costs or RNG draws."""
import ast
import copy
import inspect
import textwrap
import types


def install(module):
    original = module.run_guided_inference_visualized
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'k_step']
    assert len(loops) == 1, 'Upstream ensemble loop changed'
    loop = loops[0]
    dead_outputs = {'v_p', 'eta_p', 'gamma', 'g_dot', 's_p',
                    'score_coeff_est', 'b_drift_est', 'curr_a_sim', 'curr_t_sim'}
    changed = []
    body = []
    for node in loop.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id in dead_outputs:
            test = ast.parse('k_step < K_horizon - 1', mode='eval').body
            body.append(ast.copy_location(ast.If(test=test, body=[node], orelse=[]), node))
            changed.append(target.id)
        else:
            body.append(node)
    assert set(changed) == dead_outputs and len(changed) == 9, changed
    loop.body = body
    ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__)
    exec(compile(tree, '<my_harness_skip_unused_terminal_transition>', 'exec'), namespace)
    candidate = namespace[original.__name__]
    # Use the original module's globals so audit hooks affect both arms alike.
    module.run_guided_inference_visualized = types.FunctionType(
        candidate.__code__, module.__dict__, original.__name__, original.__defaults__)
    return dict(name='skip_unused_terminal_transition', guarded_assignments=changed,
                retains_all_cost_terms=True, retains_all_random_draws=True,
                weights_frozen=True, ensemble_size=64, ensemble_horizon=3)
