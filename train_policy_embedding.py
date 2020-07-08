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

def train_policy_embedding():
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
            
    encoder_dim = args.num_attn_heads * args.policy_attn_head_dim
    enc_input_size = env.observation_space.shape[0] + env.action_space.shape[0]

    # Initialize the Transformer encoder and decoders
    encoder = embedding_networks.make_encoder_oh(enc_input_size, N=args.num_layers, \
                                                 d_model=encoder_dim, h=args.num_attn_heads, \
                                                 dropout=args.dropout, d_emb=args.policy_embedding_dim)

    decoder = Policy(
        tuple([env.observation_space.shape[0] + args.policy_embedding_dim]),
        env.action_space,
        base_kwargs={'recurrent': False})

    embedding_networks.init_weights(encoder)
    embedding_networks.init_weights(decoder)

    encoder.train()
    decoder.train()

    encoder.to(device)
    decoder.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss(reduction='sum')

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr_policy)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr_policy)

    # Create the Environment
    env_sampler = env_utils.EnvSamplerEmb(env, all_policies, args)
    
    # Collect Train Data
    src_batch = []
    tgt_batch = []
    state_batch = []
    mask_batch = []
    mask_batch_all = []

    train_policies = [i for i in range(int(3/4*args.num_envs))]
    train_envs = [i for i in range(int(3/4*args.num_envs))]    
    # For each policy in our dataset
    for pi in train_policies:
        # For each environment in our dataset
        for env in train_envs:
            # Sample a number of trajectories for this (policy, env) pair
            for _ in range(args.num_eps_policy): 
                state_batch_t, tgt_batch_t, src_batch_t, mask_batch_t,\
                    mask_batch_all_t = env_sampler.sample_policy_data(\
                    policy_idx=pi, env_idx=env)
                state_batch.extend(state_batch_t)
                tgt_batch.extend(tgt_batch_t)
                src_batch.extend(src_batch_t)
                mask_batch.extend(mask_batch_t)
                mask_batch_all.extend(mask_batch_all_t)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch).squeeze(1)
    state_batch = torch.stack(state_batch).squeeze(1)
    mask_batch = torch.stack(mask_batch)
    mask_batch_all = torch.stack(mask_batch_all)
    num_samples_train = src_batch.shape[0]


    # Collect Eval Data
    src_batch_eval = []
    tgt_batch_eval = []
    state_batch_eval = []
    mask_batch_eval = []
    mask_batch_all_eval = []

    eval_policies = [i for i in range(int(3/4*args.num_envs))]
    eval_envs = [i for i in range(int(3/4*args.num_envs))]
    # For each policy in our dataset
    for pi in eval_policies:
        # For each environment in our dataset
        for env in eval_envs:
            # Sample a number of trajectories for this (policy, env) pair
            for _ in range(args.num_eps_policy): 
                state_batch_t, tgt_batch_t, src_batch_t, mask_batch_t, \
                    mask_batch_all_t = env_sampler.sample_policy_data(\
                    policy_idx=pi, env_idx=env)

                state_batch_eval.extend(state_batch_t)
                tgt_batch_eval.extend(tgt_batch_t)
                src_batch_eval.extend(src_batch_t)
                mask_batch_eval.extend(mask_batch_t)
                mask_batch_all_eval.extend(mask_batch_all_t)

    src_batch_eval = torch.stack(src_batch_eval).detach()
    tgt_batch_eval = torch.stack(tgt_batch_eval).squeeze(1).detach()
    state_batch_eval = torch.stack(state_batch_eval).squeeze(1).detach()
    mask_batch_eval = torch.stack(mask_batch_eval).detach()
    mask_batch_all_eval = torch.stack(mask_batch_all_eval).detach()
    num_samples_eval = src_batch_eval.shape[0]


    # Training Loop
    for epoch in range(args.num_epochs_emb + 1):
        encoder.train()
        decoder.train()

        indices = [i for i in range(num_samples_train)]
        random.shuffle(indices)
        total_counts = 0
        total_loss = 0
        num_correct_actions = 0
        for nmb in range(0, len(indices), args.policy_batch_size):
            indices_mb = indices[nmb:nmb+args.policy_batch_size]

            source = src_batch[indices_mb].to(device)
            target = tgt_batch[indices_mb].to(device)
            state = state_batch[indices_mb].to(device).float()
            mask = mask_batch[indices_mb].to(device)
            mask_all = mask_batch_all[indices_mb].squeeze(2).unsqueeze(1).to(device)

            embedding = encoder(source.detach().to(device), mask_all.detach().to(device))
            embedding = F.normalize(embedding, p=2, dim=1)

            state *= mask.to(device)
            embedding *= mask.to(device)
            recurrent_hidden_state = torch.zeros(args.policy_batch_size,
                decoder.recurrent_hidden_state_size, device=device, requires_grad=True).float()
            mask_dec = torch.zeros(args.policy_batch_size, 1, device=device, requires_grad=True).float()
            emb_state_input = torch.cat((embedding, state.to(device)), dim=1).to(device)
            action = decoder(emb_state_input, recurrent_hidden_state, mask_dec)
            action *= mask.to(device)
            target *= mask 
            
            loss = criterion(action, target.to(device))
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
        num_correct_actions_eval = 0
        for nmb in range(0, len(indices_eval), args.policy_batch_size):

            indices_mb_eval = indices_eval[nmb:nmb+args.policy_batch_size]

            source_eval = src_batch_eval[indices_mb_eval].to(device).detach()
            target_eval = tgt_batch_eval[indices_mb_eval].to(device).detach()
            state_eval = state_batch_eval[indices_mb_eval].float().to(device).detach()
            mask_eval = mask_batch_eval[indices_mb_eval].to(device).detach()
            mask_all_eval = mask_batch_all_eval[indices_mb_eval].squeeze(2).unsqueeze(1).to(device).detach()

            embedding_eval = encoder(source_eval.detach().to(device), mask_all_eval.detach().to(device)).detach()
            embedding_eval = F.normalize(embedding_eval, p=2, dim=1).detach()

            state_eval *= mask_eval.to(device).detach()
            embedding_eval *= mask_eval.to(device).detach()
            recurrent_hidden_state_eval = torch.zeros(args.policy_batch_size,
                decoder.recurrent_hidden_state_size, device='cpu').float()
            mask_dec_eval = torch.zeros(args.policy_batch_size, 1, device='cpu').float()
            emb_state_input_eval = torch.cat((embedding_eval, state_eval.to(device)), dim=1)
            action_eval = decoder(emb_state_input_eval,
                recurrent_hidden_state_eval, mask_dec_eval, deterministic=True)
            action_eval *= mask_eval.to(device)
            target_eval *= mask_eval

            loss_eval = criterion(action_eval, target_eval.to(device))

            total_loss_eval += loss_eval.item()
            total_counts_eval += len(indices_mb_eval)

        avg_loss_eval = total_loss_eval / total_counts_eval

        # Save the models
        if avg_loss_eval <= best_eval_loss:
            best_eval_loss = avg_loss_eval
            pdvf_utils.save_model("policy-encoder.", encoder, encoder_optimizer, \
                             epoch, args, args.env_name, save_dir=args.save_dir_policy_embedding)
            pdvf_utils.save_model("policy-decoder.", decoder, decoder_optimizer, \
                             epoch, args, args.env_name, save_dir=args.save_dir_policy_embedding)

        if epoch % args.log_interval == 0:
            print("# Epoch %d: Eval Loss = %.6f " % (epoch + 1, avg_loss_eval))

if __name__ == "__main__":
    train_policy_embedding()
