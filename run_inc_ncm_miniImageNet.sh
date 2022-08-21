# --- session 0 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 60 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/all_data_session0 \
--gpu 1 \
--current_session 0 \
--used_img 500 \
--balanced 0

# --- session 0 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 60 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session0 \
--gpu 1 \
--current_session 0 \
--used_img 5 \
--balanced 1

# --- session 1 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 65 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session1 \
--gpu 1 \
--current_session 1 \
--used_img 5 \
--balanced 1

# --- session 2 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 70 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session2 \
--gpu 1 \
--current_session 2 \
--used_img 5 \
--balanced 1

# --- session 3 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 75 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session3 \
--gpu 1 \
--current_session 3 \
--used_img 5 \
--balanced 1

# --- session 4 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 80 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session4 \
--gpu 1 \
--current_session 4 \
--used_img 5 \
--balanced 1

# --- session 5 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 85 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session5 \
--gpu 1 \
--current_session 5 \
--used_img 5 \
--balanced 1

# --- session 6 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 90 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session6 \
--gpu 1 \
--current_session 6 \
--used_img 5 \
--balanced 1

# --- session 7 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 95 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session7 \
--gpu 1 \
--current_session 7 \
--used_img 5 \
--balanced 1

# --- session 8 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset mini_imagenet \
--num_cls 100 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/trial1_session0_best.pth \
--exp_dir ./exp/miniImageNet/MINI-IMAGENET_base-data-2-fusion_batch-64_epoch-200_lr-1e-2_milestones-60-70_seed-1_s-30_m-4e-1/1/ncm_classifier/balanced_5_data_session8 \
--gpu 1 \
--current_session 8 \
--used_img 5 \
--balanced 1




