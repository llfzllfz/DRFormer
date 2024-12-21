save_path="./models/ArchiveII"
log_path="./log/ArchiveII"
log_path="${log_path}"
save_path="${save_path}"
filename="ArchiveII_train_5s_112"
test_filename="ArchiveII_test_5s_112"
batch_size=8
gpu=1

# for type_ in UFold SS UFold_MLDP UFold_CDP UFold_DIS_ss DIS UFold_REPEAT UFold_ss UFold_UNPAIR 

python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET_5s.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET_5s_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0
