log_path="./log/TrainSetA"
save_path="./models/TrainSetA"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi
if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
fi

log_path="${log_path}/RSS"
save_path="${save_path}/RSS"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi
if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
fi

filename="TrainSetA_112"
test_filename="TestSetA_112"

batch_size=16
epochs=100
early_stop=100
gpu=1

python train_RSS.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path "${save_path}/${filename}_SWIN_UNET2_UFold_CDP.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_UFold_CDP.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 60 \
                --batch_size $batch_size \
                --MLDP 0 --DIS 0 --CDP 1 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 0 --REPEAT 0 --UFOLD_ADD_UNPAIR 0 \
                --early_stop $early_stop \
                --epochs $epochs

python train_RSS.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path "${save_path}/${filename}_SWIN_UNET2_UFold_UNPAIR.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_UFold_UNPAIR.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 60 \
                --batch_size $batch_size \
                --MLDP 0 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 1 --REPEAT 0 --UFOLD_ADD_UNPAIR 0 \
                --early_stop $early_stop \
                --epochs $epochs

python train_RSS.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path "${save_path}/${filename}_SWIN_UNET2_UFold_REPEAT.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_UFold_REPEAT.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 60 \
                --batch_size $batch_size \
                --MLDP 0 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 0 --REPEAT 1 --UFOLD_ADD_UNPAIR 0 \
                --early_stop $early_stop \
                --epochs $epochs
