import os, random, sys
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from pdvf_networks import PDVF

from pdvf_arguments import get_args

from ppo.model import Policy
from ppo.envs import make_vec_envs

import env_utils
import pdvf_utils 
import train_utils

import myant
import myswimmer 
import myspaceship


def eval_pdvf():
    '''
    Evaluate the Policy-Dynamics Value Function. 
    '''
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    device = args.device
    if device != 'cpu':
        torch.cuda.empty_cache()

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env = make_vec_envs(args, device)
    env.reset()

    names = []
    for e in range(args.num_envs):
        for s in range(args.num_seeds):
            names.append('ppo.{}.env{}.seed{}.pt'.format(args.env_name, e, s))
            
    source_policy = []
    for name in names:
        actor_critic = Policy(
            env.observation_space.shape,
            env.action_space,
            base_kwargs={'recurrent': False})
        actor_critic.to(device)
        model = os.path.join(args.save_dir, name)
        actor_critic.load_state_dict(torch.load(model))
        source_policy.append(actor_critic)

    # Load the collected interaction episodes for each agent
    policy_encoder, policy_decoder = pdvf_utils.load_policy_model(
            args, env)
    env_encoder = pdvf_utils.load_dynamics_model(
        args, env)
 
    value_net = PDVF(env.observation_space.shape[0], args.dynamics_embedding_dim, args.hidden_dim_pdvf,
        args.policy_embedding_dim, device=device).to(device)
    value_net.to(device)
    path_to_pdvf = os.path.join(args.save_dir_pdvf, \
        "pdvf-stage{}.{}.pt".format(args.stage, args.env_name))
    value_net.load_state_dict(torch.load(path_to_pdvf)['state_dict'])
    value_net.eval()
    
    all_envs = [i for i in range(args.num_envs)]
    train_policies = [i for i in range(int(3/4*args.num_envs))]
    train_envs = [i for i in range(int(3/4*args.num_envs))]
    eval_envs = [i for i in range(int(3/4*args.num_envs), args.num_envs)]
    
    env_enc_input_size = env.observation_space.shape[0] + env.action_space.shape[0]
    sizes = pdvf_utils.DotDict({'state_dim': env.observation_space.shape[0], \
        'action_dim': env.action_space.shape[0], 'env_enc_input_size': \
        env_enc_input_size, 'env_max_seq_length': args.max_num_steps * env_enc_input_size})
    
    env_sampler = env_utils.EnvSamplerPDVF(env, source_policy, args)

    all_mean_rewards = [[] for _ in range(args.num_envs)] 
    all_mean_unnorm_rewards = [[] for _ in range(args.num_envs)] 

    # Eval on Train Envs
    train_rewards = {}
    unnorm_train_rewards = {}
    for ei in range(len(all_envs)):
        train_rewards[ei] = []
        unnorm_train_rewards[ei] = []
    for ei in train_envs:
        for i in range(args.num_eval_eps):
            args.seed = i
            np.random.seed(seed=i)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            for pi in train_policies:
                init_obs =  torch.FloatTensor(env_sampler.env.reset(env_id=ei))
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    init_state = env_sampler.env.sim.get_state()
                    res = env_sampler.zeroshot_sample_src_from_pol_state_mujoco(args, init_obs, sizes, policy_idx=pi, env_idx=ei)
                else:
                    init_state = env_sampler.env.state
                    res = env_sampler.zeroshot_sample_src_from_pol_state(args, init_obs, sizes, policy_idx=pi, env_idx=ei)

                source_env = res['source_env']
                mask_env = res['mask_env']
                source_policy = res['source_policy']
                init_episode_reward = res['episode_reward']
                mask_policy = res['mask_policy']

                if source_policy.shape[1] == 1:
                    source_policy = source_policy.repeat(1, 2, 1)
                    mask_policy = mask_policy.repeat(1, 1, 2)
                emb_policy = policy_encoder(source_policy.detach().to(device), 
                    mask_policy.detach().to(device)).detach()
                if source_env.shape[1] == 1:
                    source_env = source_env.repeat(1, 2, 1)
                    mask_env = mask_env.repeat(1, 1, 2)
                emb_env = env_encoder(source_env.detach().to(device), 
                    mask_env.detach().to(device)).detach()

                emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
                emb_env = F.normalize(emb_env, p=2, dim=1).detach()

                pred_value = value_net(init_obs.unsqueeze(0).to(device), 
                    emb_env.to(device), emb_policy.to(device)).item()
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    decoded_reward = env_sampler.get_reward_pol_embedding_state_mujoco(args,
                        init_state, init_obs, emb_policy, policy_decoder, env_idx=ei)[0]
                else:
                    decoded_reward = env_sampler.get_reward_pol_embedding_state(args, 
                        init_state, init_obs, emb_policy, policy_decoder, env_idx=ei)[0]

                qf = value_net.get_qf(init_obs.unsqueeze(0).to(device), emb_env)
                u, s, v = torch.svd(qf.squeeze())

                opt_policy_pos = u[:,0].unsqueeze(0)
                opt_policy_neg = -u[:,0].unsqueeze(0)   

                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    episode_reward_pos, num_steps_pos = env_sampler.get_reward_pol_embedding_state_mujoco(
                        args, init_state, init_obs, opt_policy_pos, policy_decoder, env_idx=ei)
                    episode_reward_neg, num_steps_neg = env_sampler.get_reward_pol_embedding_state_mujoco(
                        args, init_state, init_obs, opt_policy_neg, policy_decoder, env_idx=ei)                    
                else:
                    episode_reward_pos, num_steps_pos = env_sampler.get_reward_pol_embedding_state(
                        args, init_state, init_obs, opt_policy_pos, policy_decoder, env_idx=ei)
                    episode_reward_neg, num_steps_neg = env_sampler.get_reward_pol_embedding_state(
                        args, init_state, init_obs, opt_policy_neg, policy_decoder, env_idx=ei)

                if episode_reward_pos >= episode_reward_neg: 
                    episode_reward = episode_reward_pos
                    opt_policy = opt_policy_pos
                else:
                    episode_reward = episode_reward_neg
                    opt_policy = opt_policy_neg

                unnorm_episode_reward = episode_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_init_episode_reward = init_episode_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_decoded_reward = decoded_reward * (args.max_reward - args.min_reward) + args.min_reward
                
                unnorm_train_rewards[ei].append(unnorm_episode_reward)
                train_rewards[ei].append(episode_reward)
                if i % args.log_interval == 0:
                    if 'ant' in args.env_name or 'swimmer' in args.env_name:
                        print(f"\nTrain Environemnt: {ei} -- top singular value: {s[0].item(): .3f} --- reward after update: {unnorm_episode_reward: .3f}")
                        print(f"Initial Policy: {pi} --- init true reward: {unnorm_init_episode_reward: .3f} --- decoded: {unnorm_decoded_reward: .3f} --- predicted: {pred_value: .3f}")
                    print(f"Train Environemnt: {ei} -- top singular value: {s[0].item(): .3f} --- norm reward after update: {episode_reward: .3f}")
                    print(f"Initial Policy: {pi} --- norm init true reward: {init_episode_reward: .3f} --- norm decoded: {decoded_reward: .3f} --- predicted: {pred_value: .3f}")

        all_mean_rewards[ei].append(np.mean(train_rewards[ei]))
        all_mean_unnorm_rewards[ei].append(np.mean(unnorm_train_rewards[ei]))
                    
    for ei in train_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(all_mean_unnorm_rewards[ei]), np.std(all_mean_unnorm_rewards[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(all_mean_rewards[ei]), np.std(all_mean_rewards[ei])))


    # Eval on Eval Envs
    eval_rewards = {}
    unnorm_eval_rewards = {}
    for ei in range(len(all_envs)):
        eval_rewards[ei] = []
        unnorm_eval_rewards[ei] = []
    for ei in eval_envs:
        for i in range(args.num_eval_eps):
            args.seed = i
            np.random.seed(seed=i)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            for pi in train_policies:
                init_obs =  torch.FloatTensor(env_sampler.env.reset(env_id=ei))
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    init_state = env_sampler.env.sim.get_state()
                    res = env_sampler.zeroshot_sample_src_from_pol_state_mujoco(args, init_obs, sizes, policy_idx=pi, env_idx=ei)
                else:
                    init_state = env_sampler.env.state
                    res = env_sampler.zeroshot_sample_src_from_pol_state(args, init_obs, sizes, policy_idx=pi, env_idx=ei)

                source_env = res['source_env']
                mask_env = res['mask_env']
                source_policy = res['source_policy']
                init_episode_reward = res['episode_reward']
                mask_policy = res['mask_policy']

                if source_policy.shape[1] == 1:
                    source_policy = source_policy.repeat(1, 2, 1)
                    mask_policy = mask_policy.repeat(1, 1, 2)
                emb_policy = policy_encoder(source_policy.detach().to(device), 
                    mask_policy.detach().to(device)).detach()
                if source_env.shape[1] == 1:
                    source_env = source_env.repeat(1, 2, 1)
                    mask_env = mask_env.repeat(1, 1, 2)
                emb_env = env_encoder(source_env.detach().to(device), 
                    mask_env.detach().to(device)).detach()

                emb_policy = F.normalize(emb_policy, p=2, dim=1).detach()
                emb_env = F.normalize(emb_env, p=2, dim=1).detach()

                pred_value = value_net(init_obs.unsqueeze(0).to(device), 
                    emb_env.to(device), emb_policy.to(device)).item()
                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    decoded_reward = env_sampler.get_reward_pol_embedding_state_mujoco(args,
                        init_state, init_obs, emb_policy, policy_decoder, env_idx=ei)[0]
                else:
                    decoded_reward = env_sampler.get_reward_pol_embedding_state(args, 
                        init_state, init_obs, emb_policy, policy_decoder, env_idx=ei)[0]

                qf = value_net.get_qf(init_obs.unsqueeze(0).to(device), emb_env)
                u, s, v = torch.svd(qf.squeeze())

                opt_policy_pos = u[:,0].unsqueeze(0)
                opt_policy_neg = -u[:,0].unsqueeze(0)

                if 'ant' in args.env_name or 'swimmer' in args.env_name:
                    episode_reward_pos, num_steps_pos = env_sampler.get_reward_pol_embedding_state_mujoco(
                        args, init_state, init_obs, opt_policy_pos, policy_decoder, env_idx=ei)
                    episode_reward_neg, num_steps_neg = env_sampler.get_reward_pol_embedding_state_mujoco(
                        args, init_state, init_obs, opt_policy_neg, policy_decoder, env_idx=ei)                    
                else:
                    episode_reward_pos, num_steps_pos = env_sampler.get_reward_pol_embedding_state(
                        args, init_state, init_obs, opt_policy_pos, policy_decoder, env_idx=ei)
                    episode_reward_neg, num_steps_neg = env_sampler.get_reward_pol_embedding_state(
                        args, init_state, init_obs, opt_policy_neg, policy_decoder, env_idx=ei)
                        
                if episode_reward_pos >= episode_reward_neg: 
                    episode_reward = episode_reward_pos
                    opt_policy = opt_policy_pos
                else:
                    episode_reward = episode_reward_neg
                    opt_policy = opt_policy_neg
                
                unnorm_episode_reward = episode_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_init_episode_reward = init_episode_reward * (args.max_reward - args.min_reward) + args.min_reward
                unnorm_decoded_reward = decoded_reward * (args.max_reward - args.min_reward) + args.min_reward

                unnorm_eval_rewards[ei].append(unnorm_episode_reward)
                eval_rewards[ei].append(episode_reward)
                if i % args.log_interval == 0:
                    if 'ant' in args.env_name or 'swimmer' in args.env_name:
                        print(f"\nEval Environemnt: {ei} -- top singular value: {s[0].item(): .3f} --- reward after update: {unnorm_episode_reward: .3f}")
                        print(f"Initial Policy: {pi} --- init true reward: {unnorm_init_episode_reward: .3f} --- decoded: {unnorm_decoded_reward: .3f} --- predicted: {pred_value: .3f}")
                    print(f"Eval Environemnt: {ei} -- top singular value: {s[0].item(): .3f} --- norm reward after update: {episode_reward: .3f}")
                    print(f"Initial Policy: {pi} --- norm init true reward: {init_episode_reward: .3f} --- norm decoded: {decoded_reward: .3f} --- predicted: {pred_value: .3f}")

        all_mean_rewards[ei].append(np.mean(eval_rewards[ei]))
        all_mean_unnorm_rewards[ei].append(np.mean(unnorm_eval_rewards[ei]))
                
    for ei in train_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(all_mean_unnorm_rewards[ei]), np.std(all_mean_unnorm_rewards[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(all_mean_rewards[ei]), np.std(all_mean_rewards[ei])))

    for ei in eval_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(all_mean_unnorm_rewards[ei]), np.std(all_mean_unnorm_rewards[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(all_mean_rewards[ei]), np.std(all_mean_rewards[ei])))

    env.close()

if __name__ == '__main__':
    eval_pdvf()