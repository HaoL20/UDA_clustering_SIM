# source only
## GTA5
python tools/train_source.py --backbone "resnet101" --dataset "gta5" --num_classes 19 \
--checkpoint_dir ./log/train/GTA5_resnet101_no_CJ \
--iter_max 200000 \
--iter_stop 100000 \
--lr 2.5e-4 \
--crop_size "1280,720" \
--color_jitter no \
--numpy_transform no

python tools/train_source.py --backbone "vgg16" --dataset "gta5" --num_classes 19 \
--checkpoint_dir ./log/train/GTA5_vgg16_with_CJ \
--iter_max 200000 \
--iter_stop 100000 \
--lr 2.5e-4 \
--crop_size "1280,720" \
--color_jitter yes \
--numpy_transform no

## SYNTHIA
python tools/train_source.py --backbone "resnet101" --dataset "synthia" --num_classes 16 \
--checkpoint_dir ./log/train/source_only_SYNTHIA_resnet101_with_CJ \
--color_jitter yes \
--numpy_transform no \
--iter_max 200000 \
--iter_stop 100000 \
--lr 2.5e-4 \
--crop_size "1280,760" \

python tools/train_source.py --backbone "resnet101" --dataset "synthia" --num_classes 16 \
--checkpoint_dir ./log/train/source_only_SYNTHIA_resnet101_with_CJ \
--color_jitter yes \
--numpy_transform no \
--iter_max 200000 \
--iter_stop 100000 \
--lr 2.5e-4 \
--crop_size "1280,760" \



# UDA
python tools/train_UDA_SIM.py --source_dataset gta5 --num_classes 19 --data_loader_workers 4 --backbone resnet101 \
--round_num 5 --lr 2.5e-4 --lambda_things 1 --lambda_stuff 1 --lambda_entropy 1  --resize yes \
--checkpoint_dir ./log/UDA/GTA2city_tranGTA_thing(Entropy_1.0)_stuff(1.0)_me(Entropy_1.0)/ \
--gaussian_blur yes --color_jitter yes \
--pretrained_ckpt_file ./log/train/source_only/GTA5/resnet/GTA5_resnet101_with_CJ/gta5best.pth \
--no_uncertainty no
