filename=CDC40_HepG2
gpu=1
kfold_k=5

model=Multi_Modal
dataset=clip

second_path="./models/CLIP/${filename}"
log_paths="./log/CLIP/${filename}"


echo $second_path
echo $log_paths

if [ ! -d "$second_path" ]; then
    mkdir -p "$second_path"
fi

if [ ! -d "$log_paths" ]; then
    mkdir -p "$log_paths"
fi

python train.py --filename $filename --gpu $gpu --kfold_k $kfold_k --k_fold_data_path '/data1/llfz/Cycleformer/data/clip' \
                --metric 'AUROC' \
                $model --batch_size 32 --save_path "${second_path}/DRFormer_${kfold_k}.pt" \
                --log_path "${log_paths}/DRFormer_${kfold_k}.log" --epochs 10 --early_stop 3 \
                --dataloader 60 \
                --text_vision \
                --direction --cross \
                --cross_attention_num_layers 1 \
                --distribution 0 \
                --lr 3e-5 \
                --Vision_lr 1e-4 \
                --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
                --pretrain_module \
                --pad_length 112 \
                --eval_step -1 \
                --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar' \
                --max_step -1