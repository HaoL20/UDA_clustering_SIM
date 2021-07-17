# source only
## GTA5
python tools/train_source.py --backbone "resnet101" --dataset "gta5" --num_classes 19 \
--checkpoint_dir ./log/train/source_only_GTA5_resnet101_with_CJ \
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



