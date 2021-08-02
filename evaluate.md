# no_CJ 的numpy_transfor 一定要设置为 --numpy_transfor yes
# with_CJ:  --numpy_transfor no 

### no_CJ
python tools/evaluate.py --source_dataset "gta5" --num_classes 19 \
--backbone "resnet101" --split "test" \
--checkpoint_dir ./log/evaluate/GTA5_resnet101_noCJ/ \
--pretrained_ckpt_file ./log/train/source_only/GTA5/GTA5_resnet101_noCJ/gta5best.pth \
--save_prediction_to ./log/vis/GTA5_resnet101_noCJ/ \

python tools/evaluate.py --source_dataset "gta5" --num_classes 19 \
--backbone "vgg16" --split "test" \
--checkpoint_dir ./log/evaluate/GTA5_vgg16_noCJ/ \
--pretrained_ckpt_file ./log/train/source_only/GTA5/GTA5_vgg16_noCJ/gta5best.pth \
--save_prediction_to ./log/vis/GTA5_vgg16_noCJ/ \

# SYTHIA
python tools/evaluate.py --source_dataset "synthia" --num_classes 16 \
--backbone "resnet101" --split "test" --numpy_transfor yes \
--checkpoint_dir ./log/evaluate/SYNTHIA_resnet101_noCJ/ \
--pretrained_ckpt_file ./log/train/source_only/SYNTHIA/SYNTHIA_resnet101/synthiabest.pth \
--save_prediction_to ./log/vis/SYNTHIA_resnet101_noCJ/ 


python tools/evaluate.py --source_dataset "synthia" --num_classes 16 \
--backbone "vgg16" --split "test"  \
--checkpoint_dir ./log/evaluate/SYNTHIA_VGG16_with_CJ/ \
--pretrained_ckpt_file ./log/train/source_only/SYNTHIA/SYNTHIA_VGG16_with_CJ/synthiabest.pth \
--save_prediction_to ./log/vis/SYNTHIA_VGG16_with_CJ/ 



python tools/evaluate.py --source_dataset "synthia" --num_classes 16 \
--backbone "vgg16" --split "test" --numpy_transfor yes \
--checkpoint_dir ./log/evaluate/SYNTHIA_vgg16_noCJ/ \
--pretrained_ckpt_file ./log/train/source_only/SYNTHIA/SYNTHIA_vgg16_noCJ/synthiabest.pth \
--save_prediction_to ./log/vis/SYNTHIA_vgg16_noCJ/ 

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
