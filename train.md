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
--round_num 5 --lr 2.5e-4 --lambda_things 0.01 --lambda_stuff 0.01 --lambda_entropy 0.01  --resize yes \
--checkpoint_dir ./log/UDA/debug_GTA2city_tranGTA_thing-Squares-0.01_stuff-0.01_me-Squares-0.01_nomask \
--gaussian_blur yes --color_jitter yes \
--pretrained_ckpt_file ./log/train/source_only/GTA5/resnet/GTA5_resnet101_with_CJ/gta5best.pth \
--no_uncertainty yes



python tools/train_UDA_SIM.py --source_dataset gta5 --num_classes 19 --data_loader_workers 4 --backbone resnet101 \
--round_num 5 --lr 2.5e-4 --lambda_things 0.001 --lambda_stuff 0.001 --lambda_entropy 0.001  --resize yes \
--checkpoint_dir ./log/UDA/GTA2city_tranGTA_thing-Squares-_stuff-0.001_me-Entropy-0.001 \
--gaussian_blur yes --color_jitter yes \
--pretrained_ckpt_file ./log/train/source_only/GTA5/resnet/GTA5_resnet101_with_CJ/gta5best.pth \
--no_uncertainty yes

python tools/train_UDA_SIM.py --source_dataset gta5 --num_classes 19 --data_loader_workers 4 --backbone resnet101 \
--round_num 5 --lr 2.5e-4 --lambda_things 0.01 --lambda_stuff 0.01 --lambda_entropy 0.01  --resize yes \
--checkpoint_dir ./log/UDA/GTA2city_thing-Squares=0.01_stuff=0.01_me-Entropy=0.01_use-gta5_deeplab \
--gaussian_blur yes --color_jitter yes \
--pretrained_ckpt_file ./log/train/source_only/GTA5/resnet/GTA5_resnet101_with_CJ/gta5best.pth \
--no_uncertainty yes





python tools/train_UDA_SIM.py --source_dataset gta5 --num_classes 19 --data_loader_workers 4 --backbone resnet101 \
--round_num 5 --lr 2.5e-4 --lambda_things 0.01 --lambda_stuff 0.01 --lambda_entropy 0.01  --resize yes \
--checkpoint_dir ./log/UDA/GTA2city/FB(Entropy)_BG(0.100)_Em(Squares=0.100)_use-gta5_deeplab \
--gaussian_blur yes --color_jitter yes \
--pretrained_ckpt_file ./log/train/source_only/GTA5/resnet/GTA5_resnet101_with_CJ/gta5best.pth \
--no_uncertainty yes

FB(Squares)_BG(0.100)_Em(Entropy=0.100)_use-gta5_deeplab

python tools/train_UDA_SIM.py --source_dataset gta5 --num_classes 19 --data_loader_workers 4 --backbone resnet101 \
--round_num 5 --lr 2.5e-4 --lambda_things 0.1 --lambda_stuff 0.1 --lambda_entropy 0.1  --resize yes \
--thing_type Cosine --em_type Squares \
--checkpoint_dir ./log/UDA/GTA2city/FB\(Cosine\)_BG\(0.100\)_Em\(Squares=0.100\)_use-gta5_deeplab_right_sour_label \
--gaussian_blur yes --color_jitter yes \
--pretrained_ckpt_file ./log/train/source_only/GTA5/resnet/GTA5_resnet101_with_CJ/gta5best.pth \
--no_uncertainty yes
