set -x

export CUDA_VISIBLE_DEVICES=2,3,4,5

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/xiusic/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /shared/nas2/xiusic/ckpt/llama3-8b-rlhf-all \
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
   --prompt_data /shared/nas2/xiusic/wangyu/data/merged_data \
   --input_key context_messages,winner \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 1280 \
   --ckpt_path /shared/nas2/xiusic/ckpt/llama3-8b-rlhf-all \
   --use_wandb a4bd420f9a8eb8b97792ab04cfc06382f214dff9 \
   --rule_based_reward
