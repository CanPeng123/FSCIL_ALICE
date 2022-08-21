# --- session 0 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 100 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/all_data_session0 \
--gpu 1 \
--current_session 0 \
--used_img 500 \
--balanced 0

# --- session 0 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 100 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session0 \
--gpu 1 \
--current_session 0 \
--used_img 5 \
--balanced 1

# --- session 1 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 110 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session1 \
--gpu 1 \
--current_session 1 \
--used_img 5 \
--balanced 1

# --- session 2 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 120 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session2 \
--gpu 1 \
--current_session 2 \
--used_img 5 \
--balanced 1

# --- session 3 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 130 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session3 \
--gpu 1 \
--current_session 3 \
--used_img 5 \
--balanced 1

# --- session 4 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 140 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session4 \
--gpu 1 \
--current_session 4 \
--used_img 5 \
--balanced 1

# --- session 5 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 150 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session5 \
--gpu 1 \
--current_session 5 \
--used_img 5 \
--balanced 1

# --- session 6 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 160 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session6 \
--gpu 1 \
--current_session 6 \
--used_img 5 \
--balanced 1

# --- session 7 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 170 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session7 \
--gpu 1 \
--current_session 7 \
--used_img 5 \
--balanced 1

# --- session 8 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 180 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session8 \
--gpu 1 \
--current_session 8 \
--used_img 5 \
--balanced 1

# --- session 9 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 190 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session9 \
--gpu 1 \
--current_session 9 \
--used_img 5 \
--balanced 1

# --- session 10 ---
python main_inc_ncm.py \
--arch resnet18 \
--dataset cub200 \
--num_cls 200 \
--batch_size 32 \
--data_root ./DATA \
--pretrained ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/trial1_session0_best.pth \
--exp_dir ./exp/CUB200/CUB200_base-2-data_batch-size-512_epoch-100_lr-1e-3/1/ncm_classifier/balanced_5_data_session10 \
--gpu 1 \
--current_session 10 \
--used_img 5 \
--balanced 1


