log_path="./log/ArchiveII"
save_path="./models/ArchiveII"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi
if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
fi

batch_size=16
epochs=100
early_stop=100
gpu=2

for fold in 0 1 2 3 4
do
    type_=$fold
    filename="ArchiveII_train_${type_}_112"
    test_filename="ArchiveII_test_${type_}_112"
    python train_RSS.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                    --gpu $gpu --metric "F1" \
                    "SWIN_UNET" \
                    --save_path "${save_path}/${filename}_DRFormer_${type_}.pth" \
                    --log_path "${log_path}/${filename}_DRFormer_${type_}.log" \
                    --feature_mode 'DNAV' \
                    --dataloader_num_workers 60 \
                    --batch_size $batch_size \
                    --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
                    --early_stop $early_stop \
                    --epochs $epochs \
                    --SWIT_pretrain_path '../MAE-Lite/outputs/mae_lite/SWIN2_model_store/30.pth.tar'
done