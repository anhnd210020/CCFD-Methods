CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --rdzv_endpoint=127.0.0.1:3011 Fraud_train_Split_cc_num_2gpu.py
