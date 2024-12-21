model=Multi_Modal
dataset=$1
gpu=$2
epoch=$3
max_step=$4
eval_step=$5
lr=$6

second_path="./models/${dataset}"
log_paths="./log/${dataset}"


echo $second_path
echo $log_paths

if [ ! -d "$second_path" ]; then
    mkdir -p "$second_path"
fi

if [ ! -d "$log_paths" ]; then
    mkdir -p "$log_paths"
fi

# python train.py --gpu $gpu --k_fold_data_path "origin_data/GUE/${dataset}" --dataset "GUE" --seed 42 \
#                 --metric 'MCC' \
#                 $model --batch_size=6 --save_path "${second_path}/${model}_Text_Vision_pretrain_30_seed_0.pt" \
#                 --log_path "${log_paths}/${model}_Text_Vision_pretrain.log" --epochs $epoch --early_stop $epoch \
#                 --dataloader 30 \
#                 --text_vision \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --Vision_lr $lr \
#                 --MLDP 1 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 0 --UNPAIR 0 --REPEAT 0 --UFOLD_ADD_UNPAIR 0 \
#                 --pretrain_module \
#                 --pad_length 336 \
#                 --eval_step $eval_step \
#                 --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar' \
#                 --max_step $max_step

# python train.py --gpu $gpu --k_fold_data_path "../Cycleformer/origin_data/GUE/${dataset}" --dataset "GUE" \
#                 $model --batch_size=32 --save_path "${second_path}/${model}_Text_Vision.pt" \
#                 --log_path "${log_paths}/${model}_Text_Vision.log" --epochs $epoch --early_stop $epoch \
#                 --dataloader 30 \
#                 --text_vision \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 5e-5 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pad_length 112 \
#                 --eval_step 400
multi_gpus=2
export OMP_NUM_THREADS=$multi_gpus
python -m torch.distributed.launch --nproc_per_node=$multi_gpus --master_port=2424 --use_env train.py --gpu $gpu --k_fold_data_path "origin_data/GUE/${dataset}" --dataset "GUE" --seed 42 --multi_gpu 1 \
                --metric 'MCC' \
                $model --batch_size=8 --save_path "${second_path}/${model}_Text_Vision_pretrain_30_seed_0.pt" \
                --log_path "${log_paths}/${model}_Text_Vision_pretrain.log" --epochs $epoch --early_stop $epoch \
                --dataloader 30 \
                --text_vision \
                --direction --cross \
                --distribution 0 \
                --lr 3e-5 \
                --Vision_lr $lr \
                --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
                --pretrain_module \
                --pad_length 336 \
                --eval_step $eval_step \
                --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar' \
                --max_step $max_step




