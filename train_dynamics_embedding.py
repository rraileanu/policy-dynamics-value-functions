import os
import sys
import math
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pdvf_utils
import env_utils

import embedding_networks
from pdvf_arguments import get_args

from ppo.model import Policy
from ppo.envs import make_vec_envs

import gym

import myant
import myswimmer 
import myspaceship

def train_dynamics_embedding():
    """
    Script for training the dynamics (or environment) embeddings 
    using a transformer. 

    References:
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/
    """
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    # Useful Variables
    best_eval_loss = sys.maxsize
    device = args.device
    if device != 'cpu':
        torch.cuda.empty_cache()

    # Create the Environment
    env = make_vec_envs(args, device)

    names = []
    for e in range(args.num_envs):
        for s in range(args.num_seeds):
            names.append('ppo.{}.env{}.seed{}.pt'.format(args.env_name, e, s))
            
    all_policies = []
    for name in names:
        actor_critic = Policy(
            env.observation_space.shape,
            env.action_space,
            base_kwargs={'recurrent': False})
        actor_critic.to(device)
        model = os.path.join(args.save_dir, name)
        actor_critic.load_state_dict(torch.load(model))
        all_policies.append(actor_critic)

    encoder_dim = args.num_attn_heads * args.dynamics_attn_head_dim
    enc_input_size = 2*env.observation_space.shape[0] + env.action_space.shape[0]
    dec_input_size = args.dynamics_embedding_dim + env.observation_space.shape[0] + env.action_space.shape[0]
    output_size = env.observation_space.shape[0]

    # Create the Environment Sampler
    env_sampler = env_utils.EnvSamplerEmb(env, all_policies, args)

    # Initialize the Transformer encoder and decoders
    encoder = embedding_networks.make_encoder_oh(enc_input_size, N=args.num_layers, \
                                                 d_model=encoder_dim, h=args.num_attn_heads,\
                                                 dropout=args.dropout, d_emb=args.dynamics_embedding_dim)
    if 'spaceship' in args.env_name:
        decoder = embedding_networks.DecoderSpaceship(dec_input_size, args.dynamics_hidden_dim_cond_policy, \
            output_size, device=device)
    else:
        decoder = embedding_networks.DecoderMujoco(dec_input_size, args.dynamics_hidden_dim_cond_policy, 
            output_size, device=device)

    embedding_networks.init_weights(encoder)
    embedding_networks.init_weights(decoder)

    encoder.train()
    decoder.train()
 
    encoder.to(device)
    decoder.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss(reduction='sum')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr_dynamics)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr_dynamics)


    # Collect Train Data
    src_batch = []
    tgt_batch = []
    state_batch = []

    train_policies = [i for i in range(int(3/4*args.num_envs))]
    train_envs = [i for i in range(int(3/4*args.num_envs))]
    # For each policy in our dataset
    for pi in train_policies:
        # For each environment in our dataset
        for env in train_envs:
            for traj in range(args.num_eps_dynamics): # different environments?
                # Sample a number of trajectories for this (policy, env) pair
                state_batch_t, tgt_batch_t, src_batch_t = env_sampler.sample_k_traj_zeroshot(
                        args.num_dec_traj, policy_idx=pi, env_idx=env)
                state_batch.extend(state_batch_t)
                tgt_batch.extend(tgt_batch_t)
                src_batch.extend(src_batch_t)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    state_batch = torch.stack(state_batch)
    num_samples_train = src_batch.shape[0]


    # Collect Eval Data
    src_batch_eval = []
    tgt_batch_eval = []
    state_batch_eval = []

    eval_policies = [i for i in range(int(3/4*args.num_envs))]
    eval_envs = [i for i in range(int(3/4*args.num_envs))]
    # For each policy in our dataset
    for pi in eval_policies:
        # For each environment in our dataset
        for env in eval_envs:
            for traj in range(args.num_eps_dynamics): # different environments?
                # Sample a number of trajectories for this (policy, env) pair
                state_batch_t, tgt_batch_t, src_batch_t = env_sampler.sample_k_traj_zeroshot(
                        args.num_dec_traj, policy_idx=pi, env_idx=env)
                state_batch_eval.extend(state_batch_t)
                tgt_batch_eval.extend(tgt_batch_t)
                src_batch_eval.extend(src_batch_t)

    src_batch_eval = torch.stack(src_batch_eval).detach()
    tgt_batch_eval = torch.stack(tgt_batch_eval).detach()
    state_batch_eval = torch.stack(state_batch_eval)
    num_samples_eval = src_batch_eval.shape[0]


    # Training Loop
    for epoch in range(args.num_epochs_emb + 1):
        encoder.train()
        decoder.train()

        indices = [i for i in range(num_samples_train)]
        random.shuffle(indices)
        total_counts = 0
        total_loss = 0
        for nmb in range(0, len(indices), args.dynamics_batch_size):
            indices_mb = indices[nmb:nmb+args.dynamics_batch_size]

            source = src_batch[indices_mb]
            target = tgt_batch[indices_mb]
            state = state_batch[indices_mb]
            mask_all = (source != 0).unsqueeze(-2)
            mask_all = mask_all[:,:,:,0].squeeze(2).unsqueeze(1)

            embedding = encoder(source.detach().to(device), mask_all.detach().to(device))
            embedding = F.normalize(embedding, p=2, dim=1)

            next_state = decoder(embedding, state.to(device))
            loss = criterion(next_state, target.to(device))

            total_loss += loss.item()
            total_counts += len(indices_mb)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        if epoch % args.log_interval == 0:
            avg_loss = total_loss / total_counts
            print("\n# Epoch %d: Train Loss = %.6f " % (epoch + 1, avg_loss))


        # Evaluation
        encoder.eval()
        decoder.eval()

        indices_eval = [i for i in range(num_samples_eval)]
        total_counts_eval = 0
        total_loss_eval = 0
        for nmb in range(0, len(indices_eval), args.dynamics_batch_size):

            indices_mb_eval = indices_eval[nmb:nmb+args.dynamics_batch_size] 

            source_eval = src_batch_eval[indices_mb_eval].detach()
            target_eval = tgt_batch_eval[indices_mb_eval].detach()
            state_eval = state_batch_eval[indices_mb_eval].detach()
            mask_all_eval = (source_eval != 0).unsqueeze(-2).detach()
            mask_all_eval = mask_all_eval[:,:,:,0].squeeze(2).unsqueeze(1).detach()

            embedding_eval = encoder(source_eval.detach().to(device), mask_all_eval.detach().to(device)).detach()
            embedding_eval = F.normalize(embedding_eval, p=2, dim=1).detach()

            next_state_eval = decoder(embedding_eval, state_eval.detach().to(device)).detach()
            loss_eval = criterion(next_state_eval, target_eval.to(device)).detach()

            total_loss_eval += loss_eval.item()
            total_counts_eval += len(indices_mb_eval)

        avg_loss_eval = total_loss_eval / total_counts_eval

        # Save the models
        if avg_loss_eval <= best_eval_loss:
            best_eval_loss = avg_loss_eval
            pdvf_utils.save_model("dynamics-encoder.", encoder, encoder_optimizer, \
                             epoch + 1, args,  args.env_name, save_dir=args.save_dir_dynamics_embedding)
            pdvf_utils.save_model("dynamics-decoder.", decoder, decoder_optimizer, \
                             epoch + 1, args, args.env_name, save_dir=args.save_dir_dynamics_embedding)

        if epoch % args.log_interval == 0:
            print("# Epoch %d: Eval Loss = %.6f " % (epoch + 1, avg_loss_eval))
            

if __name__=='__main__':
    train_dynamics_embedding()