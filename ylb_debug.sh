export CUDA_VISIBLE_DEVICES='2'
#python ylb_train.py --name cyka_blyat --model texturegan --loss_texture original_image --color_space lab --ngf 64 --feature_weight 1 --pixel_weight_ab 10 --global_pixel_weight_l 10 --style_weight 100 --discriminator_weight 1 --learning_rate 2e-4 --learning_rate_D 1e-4 --data_path /backup2/Datasets/Partial_textures --batch_size 2 --visualize_every 1 --save_every 1 --num_epoch 4 --num_workers 0 --max_dataset_size 20
python ylb_train.py --name cyka_blyat --model texturegan --training_stage II --load_epoch 46 --loss_texture dtd_texture --color_space lab --ngf 64 --feature_weight 1 --pixel_weight_ab 10 --global_pixel_weight_l 10 --style_weight 100 --discriminator_weight 1 --local_pixel_weight_l 10 --discriminator_local_weight 1 --learning_rate 2e-4 --learning_rate_D 1e-4 --data_path /backup2/Datasets/Partial_textures --batch_size 1 --visualize_every 1 --save_every 1 --num_epoch 48 --num_workers 0 --max_dataset_size 20