#!/usr/bin/env python3
"""Executed linear-dynamics MPC teaching probe, not V-JEPA/robot reproduction."""
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DT = .1


def transition(state, action):
    x, v = np.asarray(state).T
    return np.stack((x + DT*v, v + DT*(np.asarray(action) - .1*v)), axis=-1)


def fit_models():
    rng = np.random.default_rng(1201)
    states = rng.uniform([-1.5, -.7], [1.5, .7], (4000, 2))
    actions = rng.uniform(-1, 1, 4000)
    targets = transition(states, actions)
    design = np.column_stack((states, actions, np.ones(len(states))))
    fitted = np.linalg.lstsq(design, targets, rcond=None)[0]
    shuffled = design.copy(); shuffled[:, 2] = rng.permutation(actions)
    broken = np.linalg.lstsq(shuffled, targets, rcond=None)[0]
    test_s = rng.uniform([-1.5, -.7], [1.5, .7], (1000, 2))
    test_a = rng.uniform(-1, 1, 1000)
    test_y = transition(test_s, test_a)
    test_d = np.column_stack((test_s, test_a, np.ones(1000)))
    return fitted, broken, {name: float(np.mean((test_d @ model-test_y)**2))
                            for name, model in [('fitted', fitted), ('shuffled_action', broken)]}


def mpc(state, goal, model, rng):
    # Random shooting, including constant sequences; identical budget for both fitted models.
    horizon, candidates = 12, 256
    actions = rng.uniform(-1, 1, (candidates, horizon))
    actions[:21] = np.linspace(-1, 1, 21)[:, None]
    predicted = np.repeat(np.asarray(state)[None], candidates, axis=0)
    costs = np.zeros(candidates)
    for k in range(horizon):
        predicted = np.column_stack((predicted, actions[:, k], np.ones(candidates))) @ model
        costs += (predicted[:, 0]-goal)**2 + .15*predicted[:, 1]**2 + .02*actions[:, k]**2
    costs += 5*(predicted[:, 0]-goal)**2 + predicted[:, 1]**2
    return float(actions[np.argmin(costs), 0])


def run():
    fitted, broken, prediction_mse = fit_models()
    rng = np.random.default_rng(1202)
    starts = rng.uniform([-.4, -.1], [.4, .1], (24, 2))
    goals = np.where(np.arange(24) % 2, 1., -1.)
    outcomes = []; trajectories = {}
    for name, model in [('PD', None), ('fitted MPC', fitted), ('shuffled-action MPC', broken)]:
        for trial, (initial, goal) in enumerate(zip(starts, goals)):
            state = initial.copy(); path = [state.tolist()]; total = 0.; limit = False
            # Same candidate stream for the two MPC systems on each paired trial.
            planner_rng = np.random.default_rng(1300+trial)
            for step in range(80):
                action = float(np.clip(2*(goal-state[0])-2.5*state[1], -1, 1)) if model is None else mpc(state, goal, model, planner_rng)
                state = transition(state, action)
                total += float((state[0]-goal)**2 + .15*state[1]**2 + .02*action**2)
                limit |= abs(state[0]) > 2
                path.append(state.tolist())
            success = abs(state[0]-goal) < .1 and abs(state[1]) < .15 and not limit
            outcomes.append({'system': name, 'trial': trial, 'initial': initial.tolist(), 'goal': float(goal),
                             'terminal_position': float(state[0]), 'terminal_velocity': float(state[1]),
                             'success': bool(success), 'out_of_bounds': bool(limit), 'cumulative_cost': total})
            if trial == 0: trajectories[name] = path
    summary = {}
    for name in ['PD', 'fitted MPC', 'shuffled-action MPC']:
        selected = [o for o in outcomes if o['system'] == name]
        summary[name] = {'successes': sum(o['success'] for o in selected), 'trials': len(selected),
                         'mean_cost': float(np.mean([o['cumulative_cost'] for o in selected])),
                         'out_of_bounds': sum(o['out_of_bounds'] for o in selected)}
    return {'scope': 'Deterministic known linear simulator and least-squares predictor; not neural video/robot evidence',
            'config': {'dt': DT, 'train_transitions': 4000, 'heldout_transitions': 1000, 'episodes': 24,
                       'episode_steps': 80, 'horizon': 12, 'candidates': 256, 'replan_every': 1,
                       'fit_seed': 1201, 'initial_state_seed': 1202, 'planning_seed': '1300 + trial',
                       'success': '|x-goal|<0.1 and |v|<0.15 at final step; never |x|>2',
                       'training_target_noise': 0, 'model_inputs': ['position', 'velocity', 'action', 'constant']},
            'models': {'fitted': fitted.tolist(), 'shuffled_action': broken.tolist()},
            'one_step_mse': prediction_mse, 'summary': summary, 'outcomes': outcomes,
            'first_trial_trajectories': trajectories}


if __name__ == '__main__':
    result = run()
    (ROOT/'sources/planning-probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'one_step_mse': result['one_step_mse'], 'summary': result['summary']}, indent=2))
