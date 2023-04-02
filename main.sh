export CUDA_VISIBLE_DEVICES=4

python -u main.py \
  --image_size 48 48 \
  --batch_size 64 \
  --class_num 7 \
  --learning_rate 0.02 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --epochs 30 \
  --milestones 20 25