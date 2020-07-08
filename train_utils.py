import torch
import torch.nn as nn
import torch.optim as optim
from pdvf_storage import TransitionPDVF, TransitionPolicyDecoder
import random

def optimize_model_pdvf(args, network, memory, num_opt_steps=1, eval=False):
    '''
    Train the Policy-Dynamics Value Function on the initial dataset (phase 1). 
    '''
    if len(memory) < args.batch_size_pdvf:
        return

    device = args.device
    value_net = network['net']
    optimizer = network['optimizer']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        transitions = memory.sample(args.batch_size_pdvf)
        batch = TransitionPDVF(*zip(*transitions))
        state_batch = torch.cat(batch.state).squeeze(1)
        emb_policy_batch = torch.cat(batch.emb_policy).squeeze(1)
        emb_env_batch = torch.cat(batch.emb_env).squeeze(1)
        total_return_batch = torch.cat(batch.total_return)
        
        state_values = value_net(state_batch.to(device).detach(), 
            emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())

        loss = l2_loss(state_values.unsqueeze(1), total_return_batch.unsqueeze(1))
        total_loss += loss.item()
        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss


def optimize_model_pdvf_phase2(args, network, memory1, memory2, num_opt_steps=1, eval=False):
    '''
    Train the Policy-Dynamics Value Function on the aggregated dataset 
    that includes both the best policy embeddings found in phase 1 
    and the original dataset (phase 2). 
    '''
    if len(memory1) < 1/4 * args.batch_size_pdvf:
        return
    if len(memory2) < 3/4 * args.batch_size_pdvf:
        return

    device = args.device
    value_net = network['net']
    optimizer = network['optimizer2']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        
        transitions1 = memory1.sample(int(1/4 * args.batch_size_pdvf))
        batch1= TransitionPDVF(*zip(*transitions1))
        state_batch1 = torch.cat(batch1.state).squeeze(1)
        emb_policy_batch1 = torch.cat(batch1.emb_policy).squeeze(1)
        emb_env_batch1 = torch.cat(batch1.emb_env).squeeze(1)
        total_return_batch1 = torch.cat(batch1.total_return)

        transitions2 = memory2.sample(int(3/4 * args.batch_size_pdvf))
        batch2= TransitionPDVF(*zip(*transitions2))
        state_batch2 = torch.cat(batch2.state).squeeze(1)
        emb_policy_batch2 = torch.cat(batch2.emb_policy).squeeze(1)
        emb_env_batch2 = torch.cat(batch2.emb_env).squeeze(1)
        total_return_batch2 = torch.cat(batch2.total_return)
        
        state_batch = torch.cat([state_batch1, state_batch2], dim=0)
        emb_policy_batch = torch.cat([emb_policy_batch1, emb_policy_batch2], dim=0)
        emb_env_batch = torch.cat([emb_env_batch1, emb_env_batch2], dim=0)
        total_return_batch = torch.cat([total_return_batch1, total_return_batch2], dim=0)

        indices = [i for i in range(state_batch.shape[0])]
        random.shuffle(indices)

        state_batch = state_batch[indices]
        emb_policy_batch = emb_policy_batch[indices]
        emb_env_batch = emb_env_batch[indices]
        total_return_batch = total_return_batch[indices]
        
        state_values = value_net(state_batch.to(device).detach(), 
            emb_env_batch.to(device).detach(), emb_policy_batch.to(device).detach())

        loss = l2_loss(state_values.unsqueeze(1), total_return_batch.unsqueeze(1))
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
 
    return total_loss


def optimize_decoder(args, network, memory, num_opt_steps=1, eval=False):
    '''
    Train the Policy Decoder on the original dataset (phase 1).
    '''
    if len(memory) < args.batch_size_pdvf:
        return

    device = args.device
    decoder = network['policy_decoder']
    optimizer = network['decoder_optimizer']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):

        transitions = memory.sample(args.batch_size_pdvf)
        batch = TransitionPolicyDecoder(*zip(*transitions))
        emb_state_batch = torch.cat(batch.emb_state).squeeze(1)
        recurrent_state_batch = torch.cat(batch.recurrent_state)
        mask_batch = torch.cat(batch.mask)
        action_batch = torch.cat(batch.action)

        pred_action =  decoder(emb_state_batch.to(device).detach(), \
                            recurrent_state_batch.to(device).detach(), \
                            mask_batch.to(device).detach())
        
        loss = l2_loss(pred_action, action_batch.detach())
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss


def optimize_decoder_phase2(args, network, memory1, memory2, num_opt_steps=1, eval=False):
    '''
    Train the Policy Decoder on the an aggregated dataset containing the 
    states and policy embeddings generated in phase 1 of training the PD-VF 
    and the original dataset (phase 2).
    '''
    if len(memory1) < 1/4 * args.batch_size_pdvf:
        return
    if len(memory2) < 3/4 * args.batch_size_pdvf:
        return

    device = args.device
    decoder = network['policy_decoder']
    optimizer = network['decoder_optimizer2']
    l2_loss = nn.MSELoss()

    total_loss = 0
    for _ in range(num_opt_steps):
        transitions1 = memory1.sample(int(1/4 * args.batch_size_pdvf))
        batch1 = TransitionPolicyDecoder(*zip(*transitions1))
        emb_state_batch1 = torch.cat(batch1.emb_state).squeeze(1)
        recurrent_state_batch1 = torch.cat(batch1.recurrent_state)
        mask_batch1 = torch.cat(batch1.mask)
        action_batch1 = torch.cat(batch1.action)
        
        transitions2 = memory2.sample(int(3/4 * args.batch_size_pdvf))
        batch2 = TransitionPolicyDecoder(*zip(*transitions2))
        emb_state_batch2 = torch.cat(batch2.emb_state).squeeze(1)
        recurrent_state_batch2 = torch.cat(batch2.recurrent_state)
        mask_batch2 = torch.cat(batch2.mask)
        action_batch2 = torch.cat(batch2.action)

        emb_state_batch = torch.cat([emb_state_batch1, emb_state_batch2], dim=0)
        recurrent_state_batch = torch.cat([recurrent_state_batch1, recurrent_state_batch2], dim=0)
        mask_batch = torch.cat([mask_batch1, mask_batch2], dim=0)
        action_batch = torch.cat([action_batch1, action_batch2], dim=0)

        indices = [i for i in range(emb_state_batch.shape[0])]
        random.shuffle(indices)

        emb_state_batch = emb_state_batch[indices]
        recurrent_state_batch = recurrent_state_batch[indices]
        mask_batch = mask_batch[indices]
        action_batch = action_batch[indices]
        
        pred_action =  decoder(emb_state_batch.to(device).detach(), \
                            recurrent_state_batch.to(device).detach(), \
                            mask_batch.to(device).detach())
        
        loss = l2_loss(pred_action, action_batch.detach())
        total_loss += loss.item()

        if not eval:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
 
    return total_loss