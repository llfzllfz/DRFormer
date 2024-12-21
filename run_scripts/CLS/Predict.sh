model=Multi_Modal
dataset=CLS
second_path="./models/${dataset}"
log_paths="./log/${dataset}"
if [ ! -d "$second_path" ]; then
    mkdir -p "$second_path"
fi

if [ ! -d "$log_paths" ]; then
    mkdir -p "$log_paths"
fi

gpu=1
# python -m torch.distributed.launch --nproc_per_node=$multi_gpus --master_port=2424 --use_env train.py --gpu $gpu --k_fold_data_path "origin_data/${dataset}" --dataset "CLS" --seed 0 --multi_gpu 1 \
#                 --metric 'MCC' \
#                 $model --batch_size=16 --save_path "${second_path}/${model}.pt" \
#                 --log_path "${log_paths}/${model}.log" --epochs 200 --early_stop 200 \
#                 --dataloader 30 \
#                 --text_vision \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --Vision_lr 1e-3 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pretrain_module \
#                 --pad_length 112 \
#                 --eval_step -1 \
#                 --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar'

# python -m torch.distributed.launch --nproc_per_node=$multi_gpus --master_port=2424 --use_env train.py --gpu $gpu --k_fold_data_path "origin_data/${dataset}" --dataset "CLS" --seed 0 --multi_gpu 1 \
#                 --metric 'MCC' \
#                 $model --batch_size=16 --save_path "${second_path}/${model}_vision.pt" \
#                 --log_path "${log_paths}/${model}_vision.log" --epochs 200 --early_stop 200 \
#                 --dataloader 30 \
#                 --vision_only \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --Vision_lr 1e-3 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pretrain_module \
#                 --pad_length 112 \
#                 --eval_step -1 \
#                 --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar'


# python -m torch.distributed.launch --nproc_per_node=$multi_gpus --master_port=2424 --use_env train.py --gpu $gpu --k_fold_data_path "origin_data/${dataset}" --dataset "CLS" --seed 0 --multi_gpu 0 \
#                 --metric 'MCC' \
#                 $model --batch_size=16 --save_path "${second_path}/${model}_text3.pt" \
#                 --log_path "${log_paths}/${model}_text3.log" --epochs 200 --early_stop 200 \
#                 --dataloader 30 \
#                 --text_only \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 1e-5 \
#                 --Vision_lr 1e-3 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pretrain_module \
#                 --pad_length 112 \
#                 --eval_step -1 \
#                 --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar'

python Predict_nRC.py --gpu $gpu --k_fold_data_path "origin_data/${dataset}" --dataset "CLS" --seed 42 --multi_gpu 0 --output 'output/nRC.csv' \
                --metric 'MCC' \
                $model --batch_size=1 --save_path "${second_path}/${model}_random.pt" \
                --log_path "${log_paths}/${model}_random.log" --epochs 200 --early_stop 200 \
                --dataloader 60 \
                --text_vision \
                --direction --cross \
                --distribution 0 \
                --lr 3e-5 \
                --Vision_lr 1e-3 \
                --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
                --pad_length 112 \
                --eval_step -1 \
                --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar'


