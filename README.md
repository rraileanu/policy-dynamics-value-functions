# Policy-Dynamics Value Functions (PD-VF) 

This is source code for the paper 

[Fast Adaptation to New Environments via Policy-Dynamics Value Functions](https://arxiv.org/pdf/2007.02879)

by Roberta Raileanu, Max Goldstein, Arthur Szlam, and Rob Fergus, 

accepted at ICML 2020. 

<img src="/figures/pdvf_gif.png" width="70%" height="70%">

## Citation
If you use this code in your own work, please cite our paper:
```
@incollection{icml2020_3993,
 abstract = {Standard RL algorithms assume fixed environment dynamics and require a significant amount of interaction to adapt to new environments. We introduce Policy-Dynamics Value Functions (PD-VF), a novel approach for rapidly adapting to dynamics different from those previously seen in training. PD-VF explicitly estimates the cumulative reward in a space of policies and environments. An ensemble of conventional RL policies is used to gather experience on training environments, from which embeddings of both policies and environments can be learned. Then, a value function conditioned on both embeddings is trained. At test time, a few actions are sufficient to infer the environment embedding, enabling a policy to be selected by maximizing the learned value function (which requires no additional environment interaction). We show that our method can rapidly adapt to new dynamics on a set of MuJoCo domains. },
 author = {Raileanu, Roberta and Goldstein, Max and Szlam, Arthur and Rob Fergus, Facebook},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {7078--7089},
 title = {Fast Adaptation to New Environments via Policy-Dynamics Value Functions},
 year = {2020}
}
```

## Requirements 
```
conda create -n pdvf python=3.7
conda activate pdvf

git clone git@github.com:rraileanu/policy-dynamics-value-functions.git
cd policy-dynamics-value-functions
pip install -r requirements.txt 

cd myant 
pip install -e .  

cd myswimmer 
pip install -e .  

cd myspaceship 
pip install -e .  
```

## (1) Reinforcement Learning Phase 

Train PPO policies on each environments, one seed for each.

Each of the commands below need to be run 
for seed in [0,...,4] and for default-ind in [0,...,19].

### Spaceship
```
python ppo/ppo_main.py \
--env-name spaceship-v0 --default-ind 0 --seed 0 
```

### Swimmer
```
python ppo/ppo_main.py \
--env-name myswimmer-v0 --default-ind 0 --seed 0 
```

### Ant-wind
```
python ppo/ppo_main.py \
--env-name myant-v0 --default-ind 0 --seed 0 
```

## (2) Self-Supervised Learning Phase

## Dynamics Embedding

### Spaceship 
```
python train_dynamics_embedding.py \
--env-name spaceship-v0 \
--dynamics-embedding-dim 8 --dynamics-batch-size 8 \
--inf-num-steps 1 --num-dec-traj 10 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings 
```

### Swimmer 
```
python train_dynamics_embedding.py \
--env-name myswimmer-v0 \
--dynamics-embedding-dim 2 --dynamics-batch-size 32  \
--inf-num-steps 1 --num-dec-traj 10 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings 
```

### Ant-wind
```
python train_dynamics_embedding.py \
--env-name myant-v0 \
--dynamics-embedding-dim 8 --dynamics-batch-size 32 \
--inf-num-steps 2 --num-dec-traj 10 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings 
```

## Policy Embedding

### Spaceship 
```
python train_policy_embedding.py \
--env-name spaceship-v0 --num-dec-traj 1 \
--save-dir-policy-embedding ./models/policy-embeddings 
```

### Swimmer 
```
python train_policy_embedding.py \
--env-name myswimmer-v0 --num-dec-traj 1 \
--save-dir-policy-embedding ./models/policy-embeddings 
```

### Ant-wind
```
python train_policy_embedding.py \
--env-name myant-v0 --num-dec-traj 1 \
--save-dir-policy-embedding ./models/policy-embeddings 
```

## (3) Supervised Learning Phase

### Spaceship 
```
python train_pdvf.py \
--env-name spaceship-v0 \
--dynamics-batch-size 8 --policy-batch-size 2048 \
--dynamics-embedding-dim 8 --policy-embedding-dim 8 \
--num-dec-traj 10 --inf-num-steps 1 --log-interval 10 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings \
--save-dir-policy-embedding ./models/policy-embeddings \
--save-dir-pdvf ./models/pdvf-models 
```

### Swimmer 
```
python train_pdvf.py \
--env-name myswimmer-v0 \
--dynamics-batch-size 8 --policy-batch-size 2048 \
--dynamics-embedding-dim 2 --policy-embedding-dim 8 \
--num-dec-traj 10 --inf-num-steps 1 --log-interval 10 \
--norm-reward --min-reward -60 --max-reward 200 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings \
--save-dir-policy-embedding ./models/policy-embeddings \
--save-dir-pdvf ./models/pdvf-models 
```

### Ant-wind
```
python train_pdvf.py \
--env-name myant-v0 \
--dynamics-batch-size 32 --policy-batch-size 2048 \
--dynamics-embedding-dim 8 --policy-embedding-dim 8 \
--num-dec-traj 10 --inf-num-steps 2 --log-interval 10 \
--norm-reward --min-reward -400 --max-reward 1000 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings \
--save-dir-policy-embedding ./models/policy-embeddings \
--save-dir-pdvf ./models/pdvf-models 
```

## (4) Evaluation Phase

### Spaceship 
```
python eval_pdvf.py \
--env-name spaceship-v0 --stage 20 \
--dynamics-batch-size 8 --policy-batch-size 2048 \
--dynamics-embedding-dim 8 --policy-embedding-dim 8 \
--num-dec-traj 10 --inf-num-steps 1 --log-interval 10 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings \
--save-dir-policy-embedding ./models/policy-embeddings \
--save-dir-pdvf ./models/pdvf-models 
```

### Swimmer 
```
python eval_pdvf.py \
--env-name myswimmer-v0 --stage 20 \
--dynamics-batch-size 8 --policy-batch-size 2048 \
--dynamics-embedding-dim 2 --policy-embedding-dim 8 \
--num-dec-traj 10 --inf-num-steps 1 --log-interval 10 \
--norm-reward --min-reward -60 --max-reward 200 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings \
--save-dir-policy-embedding ./models/policy-embeddings \
--save-dir-pdvf ./models/pdvf-models 
```

### Ant-wind
```
python eval_pdvf.py \
--env-name myant-v0 --stage 20 \
--dynamics-batch-size 32 --policy-batch-size 2048 \
--dynamics-embedding-dim 8 --policy-embedding-dim 8 \
--num-dec-traj 10 --inf-num-steps 2 --log-interval 10 \
--norm-reward --min-reward -400 --max-reward 1000 \
--save-dir-dynamics-embedding ./models/dynamics-embeddings \
--save-dir-policy-embedding ./models/policy-embeddings \
--save-dir-pdvf ./models/pdvf-models 
```

## Results 
![Performance on Test Environment](/figures/results_pdvf.png)

![Ablations](/figures/ablations_pdvf.png)


