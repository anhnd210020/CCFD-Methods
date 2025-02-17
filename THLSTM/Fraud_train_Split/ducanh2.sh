CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --rdzv_endpoint=127.0.0.1:3010 Fraud_train_Split_transactions_2gpu.py
