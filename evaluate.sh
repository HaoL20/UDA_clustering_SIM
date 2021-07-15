# GTA5

# SYTHIA
python tools/evaluate.py --source_dataset "synthia" --num_classes 16 --backbone "resnet101" --split "test" \
--checkpoint_dir ./log/evaluate/source_only_SYNTHIA_resnet101/ \
--pretrained_ckpt_file ./log/train/source_only_SYNTHIA_resnet101/synthiabest.pth \
--save_prediction_to ./log/vis/source_only_SYNTHIA_resnet101/ \



python tools/evaluate.py --source_dataset "gta5" --num_classes 19 --backbone "resnet101" --split "test" \
--resize yes --get_original_label no --pre_scale_width no \
--checkpoint_dir ./log/evaluate/syn2city-resnet_resize/ \
--pretrained_ckpt_file ./pretrained_model/synthiafinal_resnet.pth \
--save_prediction_to ./log/vis/syn2city-resnet_resize/ \
--gaussian_blur no --color_jitter no


python tools/evaluate.py --source_dataset "synthia" --num_classes 16 --backbone "resnet101" --split "test" \
--resize yes --get_original_label no --pre_scale_width no \
--checkpoint_dir ./log/evaluate/syn2city-resnet_resize/ \
--pretrained_ckpt_file ./pretrained_model/synthiafinal_resnet.pth \
--save_prediction_to ./log/vis/syn2city-resnet_resize/ \
