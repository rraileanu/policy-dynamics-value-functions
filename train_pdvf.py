import os, random, sys
import gym
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from pdvf_storage import ReplayMemoryPDVF, ReplayMemoryPolicyDecoder
from pdvf_networks import PDVF

from pdvf_arguments import get_args

from ppo.model import Policy
from ppo.envs import make_vec_envs

import env_utils
import pdvf_utils
import train_utils

import myant
import myswimmer 


def train_pdvf():
    '''
    Train the Policy-Dynamics Value Function of PD-VF
    which estimates the return for a family of policies 
    in a family of environments with varying dynamics.

    To do this, it trains a network conditioned on an initial state,
    a (learned) policy embedding, and a (learned) dynamics embedding
    and outputs an estimate of the cumulative reward of the 
    corresponding policy in the given environment. 
    '''
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    device = args.device
    if device != 'cpu':
        torch.cuda.empty_cache()

    # Create the environment
    envs = make_vec_envs(args, device)

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    names = []
    for e in range(args.num_envs):
        for s in range(args.num_seeds):
            names.append('ppo.{}.env{}.seed{}.pt'.format(args.env_name, e, s))

    all_policies = []
    for name in names:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': False})
        actor_critic.to(device)
        model = os.path.join(args.save_dir, name)
        actor_critic.load_state_dict(torch.load(model))
        all_policies.append(actor_critic)

    # Load the collected interaction episodes for each agent
    policy_encoder, policy_decoder = pdvf_utils.load_policy_model(
            args, envs)
    env_encoder = pdvf_utils.load_dynamics_model(
        args, envs)
    
    policy_decoder.train()
    decoder_optimizer = optim.Adam(policy_decoder.parameters(), lr=args.lr_policy)
    decoder_optimizer2 = optim.Adam(policy_decoder.parameters(), lr=args.lr_policy)
    decoder_network = {'policy_decoder': policy_decoder, \
                        'decoder_optimizer': decoder_optimizer, \
                        'decoder_optimizer2': decoder_optimizer2}
    
    # Instantiate the PD-VF, Optimizer and Loss
    args.use_l2_loss = True
    value_net = PDVF(envs.observation_space.shape[0], args.dynamics_embedding_dim, args.hidden_dim_pdvf,
        args.policy_embedding_dim, device=device).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=args.lr_pdvf, eps=args.eps)
    optimizer2 = optim.Adam(value_net.parameters(), lr=args.lr_pdvf, eps=args.eps)
    network = {'net': value_net, 'optimizer': optimizer, 'optimizer2': optimizer2}
    value_net.train()

    train_policies = [i for i in range(int(3/4*args.num_envs))]
    train_envs = [i for i in range(int(3/4*args.num_envs))]
    eval_envs = [i for i in range(int(3/4*args.num_envs), args.num_envs)]

    all_envs = [i for i in range(args.num_envs)]
    
    NUM_STAGES = args.num_stages
    NUM_TRAIN_EPS = args.num_train_eps
    NUM_TRAIN_SAMPLES = NUM_TRAIN_EPS * len(train_policies) * len(train_envs)
    NUM_EVAL_EPS = args.num_eval_eps
    NUM_EVAL_SAMPLES = NUM_EVAL_EPS * len(train_policies) * len(train_envs)

    env_enc_input_size = 2*envs.observation_space.shape[0] + args.policy_embedding_dim
    sizes = pdvf_utils.DotDict({'state_dim': envs.observation_space.shape[0], \
        'action_dim': envs.action_space.shape[0], 'env_enc_input_size': env_enc_input_size, \
        'env_max_seq_length': args.max_num_steps * env_enc_input_size})
    
    env_sampler = env_utils.EnvSamplerPDVF(envs, all_policies, args)
    decoder_env_sampler = env_utils.EnvSamplerEmb(envs, all_policies, args)


    ####################    TRAIN PHASE 1      ########################
    # Collect Eval Data for First Training Stage
    eval_memory = ReplayMemoryPDVF(NUM_EVAL_SAMPLES)
    decoder_eval_memory = ReplayMemoryPolicyDecoder(NUM_EVAL_SAMPLES)
    for i in range(NUM_EVAL_EPS):
        for ei in train_envs:
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
                mask_policy = res['mask_policy']
                source_policy = res['source_policy']
                episode_reward = res['episode_reward']
                episode_reward_tensor = torch.tensor([episode_reward], 
                    device=device, dtype=torch.float)

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
                decoded_reward_tensor = torch.tensor([decoded_reward], 
                    device=device, dtype=torch.float)

                eval_memory.push(init_obs.unsqueeze(0), emb_policy.unsqueeze(0),
                    emb_env.unsqueeze(0), episode_reward_tensor)

                # Collect data for the decoder
                state_batch, tgt_batch, src_batch, mask_batch, _ = \
                    decoder_env_sampler.sample_policy_data(policy_idx=pi, env_idx=ei)

                for state, tgt, src, mask in zip(state_batch, tgt_batch, src_batch, mask_batch):
                    state = state.to(device).float()
                    mask = mask.to(device)
                    
                    state *= mask.to(device).detach()
                    emb_policy *= mask.to(device).detach()
                    recurrent_state = torch.zeros(state.shape[0],
                        policy_decoder.recurrent_hidden_state_size, device=args.device).float()
                    mask_dec = torch.zeros(state.shape[0], 1, device=args.device).float()
                    emb_state = torch.cat((emb_policy, state.to(device)), dim=1)
                    action = policy_decoder(emb_state, recurrent_state, mask_dec, deterministic=True)
                    action *= mask.to(device)
                    decoder_eval_memory.push(emb_state, recurrent_state, mask_dec, action)


    # Collect Train Data for Frist Training Stage
    memory = ReplayMemoryPDVF(NUM_TRAIN_SAMPLES)
    decoder_memory = ReplayMemoryPolicyDecoder(NUM_TRAIN_SAMPLES)
    for i in range(NUM_TRAIN_EPS):
        for ei in train_envs:
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
                mask_policy = res['mask_policy']
                episode_reward = res['episode_reward']
                
                episode_reward_tensor = torch.tensor([episode_reward], 
                    device=device, dtype=torch.float)

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
                decoded_reward_tensor = torch.tensor([decoded_reward], 
                    device=device, dtype=torch.float)

                memory.push(init_obs.unsqueeze(0), emb_policy.unsqueeze(0),
                    emb_env.unsqueeze(0), episode_reward_tensor)

                # Collect data for the decoder
                state_batch, tgt_batch, src_batch, mask_batch, _ = \
                    decoder_env_sampler.sample_policy_data(policy_idx=pi, env_idx=ei)

                for state, tgt, src, mask in zip(state_batch, tgt_batch, src_batch, mask_batch):
                    state = state.to(device).float()
                    mask = mask.to(device)
                    
                    state *= mask.to(device).detach()
                    emb_policy *= mask.to(device).detach()
                    recurrent_state = torch.zeros(state.shape[0],
                        policy_decoder.recurrent_hidden_state_size, device=args.device).float()
                    mask_dec = torch.zeros(state.shape[0], 1, device=args.device).float()
                    emb_state = torch.cat((emb_policy, state.to(device)), dim=1)
                    action = policy_decoder(emb_state, recurrent_state, mask_dec, deterministic=True)
                    action *= mask.to(device)
                    decoder_memory.push(emb_state, recurrent_state, mask_dec, action)


    ### Train - Stage 1 ###
    total_train_loss = 0
    total_eval_loss = 0
    BEST_EVAL_LOSS = sys.maxsize 

    decoder_total_train_loss = 0
    decoder_total_eval_loss = 0
    DECODER_BEST_EVAL_LOSS = sys.maxsize
    print("\nFirst Training Stage")
    for i in range(args.num_epochs_pdvf_phase1):
        train_loss = train_utils.optimize_model_pdvf(args, network,
            memory, num_opt_steps=args.num_opt_steps)
        if train_loss:
            total_train_loss += train_loss  

        eval_loss = train_utils.optimize_model_pdvf(args, network,
            eval_memory, num_opt_steps=args.num_opt_steps, eval=True)

        if eval_loss:
            total_eval_loss += eval_loss
            if eval_loss < BEST_EVAL_LOSS:
                BEST_EVAL_LOSS = eval_loss
                pdvf_utils.save_model("pdvf-stage0.", value_net, optimizer, \
                                i, args, args.env_name, save_dir=args.save_dir_pdvf)

        if i % args.log_interval == 0:
            print("\n### PD-VF: Episode {}: Train Loss {:.6f} Eval Loss {:.6f}".format( \
                i, total_train_loss / args.log_interval, total_eval_loss / args.log_interval))
            total_train_loss = 0
            total_eval_loss = 0
        
        # Train the Policy Decoder on mixed data 
        # from trajectories collected using the pretrained policies 
        # and decoded trajectories by the current decoder
        decoder_train_loss = train_utils.optimize_decoder(args, decoder_network,
            decoder_memory, num_opt_steps=args.num_opt_steps)
        if decoder_train_loss:
            decoder_total_train_loss += decoder_train_loss

        decoder_eval_loss = train_utils.optimize_decoder(args, decoder_network,
            decoder_eval_memory, num_opt_steps=args.num_opt_steps, eval=True)           

        if decoder_eval_loss:
            decoder_total_eval_loss += decoder_eval_loss
        
            if decoder_eval_loss < DECODER_BEST_EVAL_LOSS:
                DECODER_BEST_EVAL_LOSS = decoder_eval_loss
                pdvf_utils.save_model("policy-decoder-stage0.", policy_decoder, decoder_optimizer, \
                                i, args, args.env_name, save_dir=args.save_dir_pdvf)
        
        if i % args.log_interval == 0:
            print("### PolicyDecoder: Episode {}: Train Loss {:.6f} Eval Loss {:.6f}".format( \
                i, decoder_total_train_loss / args.log_interval, decoder_total_eval_loss / args.log_interval))
            decoder_total_train_loss = 0
            decoder_total_eval_loss = 0
            

    ####################    TRAIN PHASE 2    ########################
    for k in range(NUM_STAGES):
        print("Stage in Second Training Phase: ", k)
        # Collect  Eval Data for Second Training Stage
        eval_memory2 = ReplayMemoryPDVF(NUM_EVAL_SAMPLES)
        decoder_eval_memory2 = ReplayMemoryPolicyDecoder(NUM_EVAL_SAMPLES)
        for i in range(NUM_EVAL_EPS):
            for ei in train_envs:
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
                    mask_policy = res['mask_policy']
                    init_episode_reward = res['episode_reward']

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

                    
                    episode_reward_tensor = torch.tensor([episode_reward], device=device, dtype=torch.float)

                    eval_memory2.push(init_obs.unsqueeze(0), opt_policy.unsqueeze(0),
                        emb_env.unsqueeze(0), episode_reward_tensor)

                    
                    if 'ant' in args.env_name or 'swimmer' in args.env_name:
                        all_emb_state, all_recurrent_state, all_mask, all_action = \
                            decoder_env_sampler.get_decoded_traj_mujoco(args, init_state, \
                                init_obs, opt_policy_pos, policy_decoder, env_idx=ei)
                    else:
                        all_emb_state, all_recurrent_state, all_mask, all_action = \
                            decoder_env_sampler.get_decoded_traj(args, init_state, \
                                init_obs, opt_policy_pos, policy_decoder, env_idx=ei)

                    for e, r, m, a in zip(all_emb_state, all_recurrent_state, all_mask, all_action):
                        decoder_eval_memory2.push(e, r, m, a)


        # Collect  Train Data for Second Training Stage
        memory2 = ReplayMemoryPDVF(NUM_TRAIN_SAMPLES)
        decoder_memory2 = ReplayMemoryPolicyDecoder(NUM_TRAIN_SAMPLES)
        for i in range(NUM_TRAIN_EPS):
            for ei in train_envs:
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
                    mask_policy = res['mask_policy']
                    init_episode_reward = res['episode_reward']

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

                    qf = value_net.get_qf(init_obs.unsqueeze(0).to(device), emb_env)
                    u, s, v = torch.svd(qf.squeeze())
                    
                    opt_policy_pos = u[:,0].unsqueeze(0)
                    opt_policy_neg = -u[:,0].unsqueeze(0)

                    # include both solutions (positive and negative policy emb) in the train data 
                    # to enforce the correct shape and make it aware of the two
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
                    
                    episode_reward_tensor_pos = torch.tensor([episode_reward_pos], device=device, dtype=torch.float)
                    episode_reward_tensor_neg = torch.tensor([episode_reward_neg], device=device, dtype=torch.float)

                    memory2.push(init_obs.unsqueeze(0), opt_policy_pos.unsqueeze(0),
                        emb_env.unsqueeze(0), episode_reward_tensor_pos)
                    memory2.push(init_obs.unsqueeze(0), opt_policy_neg.unsqueeze(0),
                        emb_env.unsqueeze(0), episode_reward_tensor_neg)

                    # collect PolicyDecoder train data for second training stage
                    if 'ant' in args.env_name or 'swimmer' in args.env_name:
                        all_emb_state, all_recurrent_state, all_mask, all_action = \
                            decoder_env_sampler.get_decoded_traj_mujoco(args, init_state, \
                                init_obs, opt_policy_pos, policy_decoder, env_idx=ei)
                    else:
                        all_emb_state, all_recurrent_state, all_mask, all_action = \
                            decoder_env_sampler.get_decoded_traj(args, init_state, \
                                init_obs, opt_policy_pos, policy_decoder, env_idx=ei)

                    for e, r, m, a in zip(all_emb_state, all_recurrent_state, all_mask, all_action):
                        decoder_memory2.push(e, r, m, a)

        ### Train - Stage 2 ###
        total_train_loss = 0
        total_eval_loss = 0
        BEST_EVAL_LOSS = sys.maxsize
        
        decoder_total_train_loss = 0
        decoder_total_eval_loss = 0
        DECODER_BEST_EVAL_LOSS = sys.maxsize
        for i in range(args.num_epochs_pdvf_phase2):
            train_loss = train_utils.optimize_model_pdvf_phase2(args, network,
                memory, memory2, num_opt_steps=args.num_opt_steps)
            if train_loss:
                total_train_loss += train_loss

            eval_loss = train_utils.optimize_model_pdvf_phase2(args, network,
                eval_memory, eval_memory2, num_opt_steps=args.num_opt_steps,
                eval=True)        

            if eval_loss:   
                total_eval_loss += eval_loss
            
                if eval_loss < BEST_EVAL_LOSS:
                    BEST_EVAL_LOSS = eval_loss
                    pdvf_utils.save_model("pdvf-stage{}.".format(k+1), value_net, optimizer, \
                                    i, args, args.env_name, save_dir=args.save_dir_pdvf)

            if i % args.log_interval == 0:
                print("\n### PDVF: Stage {} -- Episode {}: Train Loss {:.6f} Eval Loss {:.6f}".format( \
                    k, i, total_train_loss / args.log_interval, total_eval_loss / args.log_interval))
                total_train_loss = 0
                total_eval_loss = 0
        
            # Train the Policy Decoder on mixed data 
            # from trajectories collected using the pretrained policies 
            # and decoded trajectories by the current decoder
            decoder_train_loss = train_utils.optimize_decoder_phase2(args, decoder_network,
                decoder_memory, decoder_memory2, num_opt_steps=args.num_opt_steps)
            if decoder_train_loss:
                decoder_total_train_loss += decoder_train_loss

            decoder_eval_loss = train_utils.optimize_decoder_phase2(args, decoder_network,
                decoder_eval_memory, decoder_eval_memory2, num_opt_steps=args.num_opt_steps,
                eval=True)            
            if decoder_eval_loss:
                decoder_total_eval_loss += decoder_eval_loss
            
                if decoder_eval_loss < DECODER_BEST_EVAL_LOSS:
                    DECODER_BEST_EVAL_LOSS = decoder_eval_loss
                    pdvf_utils.save_model("policy-decoder-stage{}.".format(k+1), policy_decoder, decoder_optimizer, \
                                    i, args, args.env_name, save_dir=args.save_dir_pdvf)

            if i % args.log_interval == 0:
                print("### PolicyDecoder: Stage {} -- Episode {}: Train Loss {:.6f} Eval Loss {:.6f}".format( \
                    k, i, decoder_total_train_loss / args.log_interval, decoder_total_eval_loss / args.log_interval))
                decoder_total_train_loss = 0
                decoder_total_eval_loss = 0


    ####################         EVAL        ########################

    # Eval on Train Envs
    value_net.eval()
    policy_decoder.eval()
    train_rewards = {}
    unnorm_train_rewards = {}
    for ei in range(len(all_envs)):
        train_rewards[ei] = []
        unnorm_train_rewards[ei] = []
    for i in range(NUM_EVAL_EPS):
        for ei in train_envs:
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
                    
    for ei in train_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(unnorm_train_rewards[ei]), np.std(unnorm_train_rewards[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(train_rewards[ei]), np.std(train_rewards[ei])))

    # Eval on Eval Envs
    value_net.eval()
    policy_decoder.eval()
    eval_rewards = {}
    unnorm_eval_rewards = {}
    for ei in range(len(all_envs)):
        eval_rewards[ei] = []
        unnorm_eval_rewards[ei] = []

    for i in range(NUM_EVAL_EPS):
        for ei in eval_envs:
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
                    
    for ei in train_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(unnorm_train_rewards[ei]), np.std(unnorm_train_rewards[ei])))
        else:
            print("Train Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(train_rewards[ei]), np.std(train_rewards[ei])))
       
    for ei in eval_envs:
        if 'ant' in args.env_name or 'swimmer' in args.env_name:
            print("Eval Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(unnorm_eval_rewards[ei]), np.std(unnorm_eval_rewards[ei])))
        else:
             print("Eval Env {} has reward with mean {:.3f} and std {:.3f}"\
                .format(ei, np.mean(eval_rewards[ei]), np.std(eval_rewards[ei])))

    envs.close()

if __name__ == "__main__":
    train_pdvf()