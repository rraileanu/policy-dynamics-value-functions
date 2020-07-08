import argparse
import torch
import math
import sys

def get_args():
    parser = argparse.ArgumentParser(description='PDVF')

    # PPO
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='number of CPU processes to use for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size used for training the embeddings')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save interval, one checkpoint per n updates')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon')
    parser.add_argument('--log-dir', default='./logs',
                        help='directory to save agent logs (..usfa-marl-data/logs)')
    parser.add_argument('--save-dir', default='./models/ppo-policies/',
                        help='directory to save models')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--env-name', type=str, default='spaceship-v0', help='environment')
    parser.add_argument('--basepath', type=str, 
                        default='/home/roberta/miniconda3/envs/pdvf/lib/python3.7/site-packages/gym/envs/mujoco/assets/',
                        help='path to mujoco xml files')    
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value to clip gradient norm')
    
    # PDVF
    parser.add_argument('--stage', type=int, default=20,
                        help='stage of training for evaluating the PDVF')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='normalize the reward when training the PDVF to be between 0 and 1 \
                        -- needed for the quadratic optimization')
    parser.add_argument('--min-reward', type=float, default=-400,
                        help='minimum reward used for normalization')
    parser.add_argument('--max-reward', type=float, default=1000,
                        help='maximum reward used for normalization')
    parser.add_argument('--dynamics-batch-size', type=int, default=32,
                        help='batch size used for training the dynamics embedding')
    parser.add_argument('--policy-batch-size', type=int, default=2048,
                        help='batch size used for training the policy embedding')
    parser.add_argument('--inf-num-steps', type=int, default=1, 
                        help='number of interactions with the new environment \
                        used to infer the dynamics embeddding')
    parser.add_argument('--num-train-eps', type=int, default=1, 
                        help='number of training episodes for the embeddings')
    parser.add_argument('--num-eval-eps', type=int, default=1, 
                        help='number of evaluation episodes for the embeddings')
    parser.add_argument('--num-stages', type=int, default=20, 
                        help='number of stages for training the PDVF')
    parser.add_argument('--num-epochs-pdvf-phase1', type=int, default=200, 
                        help='number of epochs for training the PDVF in phase 1')
    parser.add_argument('--num-epochs-pdvf-phase2', type=int, default=100,
                        help='number of epochs for training the PDVF in phase 2')
    parser.add_argument('--num-envs', type=int, default=20,
                        help='total number of environments (both train and test)')
    parser.add_argument('--default-ind', type=int, default=0,
                        help='default index for the train envs')
    parser.add_argument('--lr-dynamics', type=float, default=0.001,
                        help='learning rate for the dynamics embedding')  
    parser.add_argument('--lr-policy', type=float, default=0.01,
                        help='learning rates for the policy embedding')
    parser.add_argument('--num-eps-dynamics', type=int, default=100,
                        help='number of episodes used to train the dynamics embedding')  
    parser.add_argument('--num-eps-policy', type=int, default=200,
                        help='number of episodes used to train the policy embedding')  
    parser.add_argument('--lr-pdvf', type=float, default=0.005,
                        help='learning rate for training the PDVF')
    parser.add_argument('--save-dir-policy-embedding', \
                        default='./models/policy-embeddings/',
                        help='directory to save models the policy embedding models')
    parser.add_argument('--save-dir-dynamics-embedding', \
                        default='./models/dynamics-embeddings/',
                        help='directory to save models the dynamics embedding models')
    parser.add_argument('--save-dir-pdvf', \
                        default='../gvf_logs/pdvf-policies/',
                        help='directory to save models the PDVF')
    parser.add_argument('--batch-size-pdvf', type=int, default=128,
                        help='batch size used for training the PDVF')
    parser.add_argument('--hidden-dim-pdvf', type=int, default=128,
                        help='dimension of the hidden layers for the PDVF network')
    parser.add_argument('--num-opt-steps', type=int, default=1,
                        help='number of optimization steps to \
                        find the optimal policy of the PDVF')
    parser.add_argument('--num-seeds', type=int, default=5, 
                        help='number of seeds used to collect \
                        PPO policies in all the environments')
    
    # Embeddings
    parser.add_argument('--num-epochs-emb', type=int, default=200,
                        help='number of epochs for training the embeddings')
    parser.add_argument('--num-dec-traj', type=int, default=10,
                        help='number of trajectories to train the embeddings')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of layers in the encoder (with self attention)')
    parser.add_argument('--num-attn-heads', type=int, default=1,
                        help='number of attention heads')    
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout for use with rnn + attention decoder in seq2seq \
                        embedding model')    

    # Spaceship 
    parser.add_argument('--max-num-steps', type=int, default=50,
                        help='maximum number of steps allowed in the environment')
    parser.add_argument('--policy-embedding-dim', type=int, default=8,
                        help='dimension of the policy embedding')
    parser.add_argument('--dynamics-embedding-dim', type=int, default=64,
                        help='dimension of the dynamics embedding')
    parser.add_argument('--policy-hidden-dim-cond-policy', type=int, default=32,
                        help='hidden dimension of the policy autoencoder')
    parser.add_argument('--dynamics-hidden-dim-cond-policy', type=int, default=32,
                        help='hidden dimension of the environment autoencoder')
    parser.add_argument('--policy-attn-head-dim', type=int, default=64,
                        help='dimension of the policy attention head')
    parser.add_argument('--dynamics-attn-head-dim', type=int, default=64,
                        help='dimension of the dynamics attention head')
    
    # parse all arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
