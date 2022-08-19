python main_base.py \
--dataset cifar100 \
--data_root ./DATA \
--exp_dir ./exp/CIFAR100_base-data-2-fusion_cosFace_batch-128_epoch-100_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1 \
--arch resnet18 \
--learning_rate 0.01 \
--epochs_base 100 \
--epochs_new 100 \
--weight_decay 5e-4 \
--momentum 0.9 \
--milestones 60 70 \
--gamma 0.1 \
--batch_size 128 \
--eval_freq 1 \
--save_freq 25 \
--feat_dim 2048 \
--loss_type cosface \
--loss_s 30 \
--loss_m 0.4 \
--seed 1 \
--data_transform \
--data_fusion 


