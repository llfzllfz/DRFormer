model=Multi_Modal
dataset=$1
gpu=$2

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

# python Evaluate.py --gpu $gpu --k_fold_data_path "origin_data/GUE/${dataset}" --dataset "GUE" --seed 42 \
#                 $model --batch_size=32 --save_path "${second_path}/${model}_Text_only.pt" \
#                 --log_path "${log_paths}/${model}_Text_only_Evaluate.log" --epochs 30 --early_stop 5 \
#                 --dataloader 60 \
#                 --text_only \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pad_length 112

# python Evaluate.py --gpu $gpu --k_fold_data_path "origin_data/GUE/${dataset}" --dataset "GUE" --seed 42 \
#                 $model --batch_size=32 --save_path "${second_path}/${model}_Text_only_pretrain.pt" \
#                 --log_path "${log_paths}/${model}_Text_only_Evaluate.log" --epochs 30 --early_stop 5 \
#                 --dataloader 60 \
#                 --text_only \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pad_length 112

# python Evaluate.py --gpu $gpu --k_fold_data_path "../Cycleformer/origin_data/GUE/${dataset}" --dataset "GUE" \
#                 $model --batch_size=32 --save_path "${second_path}/${model}_Vision_only.pt" \
#                 --log_path "${log_paths}/${model}_Vision_only_Evaluate.log" --epochs 30 --early_stop 5 \
#                 --dataloader 30 \
#                 --vision_only \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pad_length 112

# python Evaluate.py --gpu $gpu --k_fold_data_path "origin_data/GUE/${dataset}" --dataset "GUE" --seed 42 \
#                 $model --batch_size=32 --save_path "${second_path}/${model}_Vision_only_pretrain_30_seed_42.pt" \
#                 --log_path "${log_paths}/${model}_Vision_only_Evaluate.log" --epochs 30 --early_stop 5 \
#                 --dataloader 30 \
#                 --vision_only \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pad_length 112

# python Evaluate.py --gpu $gpu --k_fold_data_path "../Cycleformer/origin_data/GUE/${dataset}" --dataset "GUE" \
#                 $model --batch_size=32 --save_path "${second_path}/${model}_Text_Vision.pt" \
#                 --log_path "${log_paths}/${model}_Text_Vision_Evaluate.log" --epochs 30 --early_stop 5 \
#                 --dataloader 30 \
#                 --text_vision \
#                 --direction --cross \
#                 --distribution 0 \
#                 --lr 3e-5 \
#                 --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
#                 --pad_length 112

python Evaluate.py --gpu $gpu --k_fold_data_path "origin_data/GUE/${dataset}" --dataset "GUE" --seed 42 \
                $model --batch_size=32 --save_path "${second_path}/${model}_Text_Vision_pretrain_30_seed_0.pt" \
                --log_path "${log_paths}/${model}_Text_Vision_Evaluate.log" --epochs 30 --early_stop 5 \
                --dataloader 30 \
                --text_vision \
                --direction --cross \
                --distribution 0 \
                --lr 3e-5 \
                --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
                --pad_length 112
