import argparse
import torch as pt

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--act-fn', default='relu', help='core activation function (default: relu)')
    parser.add_argument('--batch-size', default=int(16), help='batch size (default: 16)')
    parser.add_argument('--d-cue', default=int(1), help='cue dimension (default: 1)')
    parser.add_argument('--device', default='cpu', help='device (default: cpu)')
    parser.add_argument('--discount', default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--env-id', default='None', help='environment id (default: None)')
    parser.add_argument('--exploration-noise', default=1.0, help='standard deviation of exploration noise (default: 1.0)')
    parser.add_argument('--grad', default='sparse', help='gradient sparsity (default: sparse)')
    parser.add_argument('--height', default=int(5), help='maze height (default: 5)')
    parser.add_argument('--k-B', default=1.0, help='sampling distribution temperature (default: 1.0)')
    parser.add_argument('--len-cue-sequence', default=int(20), help='length input sequence (default: 20)')
    parser.add_argument('--log-iteration', default=int(1e3), help='log at regular intervals (default: 1000)')
    parser.add_argument('--loss-fn', default='mse', help='loss function (default: mse)')
    parser.add_argument('--lr', default=3e-3, help='learning rate (default: 3e-3)')
    parser.add_argument('--momentum', default=0.0, help='momentum (default: 0.0)')
    parser.add_argument('--n-actions', default=int(0), help='number of actions, depending on env (default: 0)')
    parser.add_argument('--n-cues', default=int(2), help='number of cues (default: 2)')
    parser.add_argument('--n-episodes', default=int(1e4), help='number of training episodes (default: 1e4)')
    parser.add_argument('--n-hidden-neurons', default=int(20), help='number of hidden neurons (default: 20)')
    parser.add_argument('--n-layers', default=int(2), help='network layers (default: 2)')
    parser.add_argument('--n-steps', default=int(20), help='number environment steps until done (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--optimizer-class', default='adam', help='optimizer for training (default: adam)')
    parser.add_argument('--save-iteration', default=int(1e3), help='save at regular intervals (default: 1000)')
    parser.add_argument('--seed', default=False, help='random seed (default: False)')
    parser.add_argument('--std', default=1e-1, help='standard deviation for evolution strategy (default: 1e-1)')
    parser.add_argument('--stochastic', default=True, help='transition sampling stochastic (default: True)')
    parser.add_argument('--tau', default=0.005, help='gradient decay rate for target networks (default: 0.005)')
    parser.add_argument('--weight-decay', default=0.0, help='L2 penalty (default: 0.0)')

    args = parser.parse_args(args=[])
    args.cuda = not args.no_cuda and pt.cuda.is_available()

    return args