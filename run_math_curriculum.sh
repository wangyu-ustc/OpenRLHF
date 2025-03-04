set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --save_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l1 \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /shared/nas2/xiusic/wangyu/data/math_data_level1 \
   --input_key context_messages,labels \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 1280 \
   --ckpt_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l1 \
   --use_wandb a4bd420f9a8eb8b97792ab04cfc06382f214dff9 \
   --rule_based_reward


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l1 \
   --save_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l2 \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /shared/nas2/xiusic/wangyu/data/math_data_level2 \
   --input_key context_messages,labels \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 1280 \
   --ckpt_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l2 \
   --use_wandb a4bd420f9a8eb8b97792ab04cfc06382f214dff9 \
   --rule_based_reward


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l2 \
   --save_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l3 \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /shared/nas2/xiusic/wangyu/data/math_data_level3 \
   --input_key context_messages,labels \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 1280 \
   --ckpt_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l3 \
   --use_wandb a4bd420f9a8eb8b97792ab04cfc06382f214dff9 \
   --rule_based_reward


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l3 \
   --save_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l4 \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /shared/nas2/xiusic/wangyu/data/math_data_level4 \
   --input_key context_messages,labels \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 1280 \
   --ckpt_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l4 \
   --use_wandb a4bd420f9a8eb8b97792ab04cfc06382f214dff9 \
   --rule_based_reward


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l4 \
   --save_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l5 \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /shared/nas2/xiusic/wangyu/data/math_data_level5 \
   --input_key context_messages,labels \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 1280 \
   --ckpt_path /shared/nas2/xiusic/ckpt/qwen-7b-instruct-math-l5 \
   --use_wandb a4bd420f9a8eb8b97792ab04cfc06382f214dff9 \
   --rule_based_reward
