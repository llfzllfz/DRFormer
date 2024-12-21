save_path="./models/ArchiveII"
log_path="./log/ArchiveII"
log_path="${log_path}/RSS"
save_path="${save_path}/RSS"
filename="ArchiveII_train_112"
test_filename="ArchiveII_test_112"
batch_size=8
gpu=1

# for type_ in UFold SS UFold_MLDP UFold_CDP UFold_DIS_ss DIS UFold_REPEAT UFold_ss UFold_UNPAIR 
type_="UFold"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 0 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 0 --REPEAT 0 --UFOLD_ADD_UNPAIR 0

type_="SS"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 1 --DIS 0 --CDP 1 --SPTIAL_DIS 0 --UFOLD 0 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0

type_="UFold_MLDP"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 1 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 0 --REPEAT 0 --UFOLD_ADD_UNPAIR 0

type_="UFold_CDP"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 0 --DIS 0 --CDP 1 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 0 --REPEAT 0 --UFOLD_ADD_UNPAIR 0

type_="UFold_UNPAIR"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 0 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 1 --REPEAT 0 --UFOLD_ADD_UNPAIR 0

type_="UFold_REPEAT"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 0 --DIS 0 --CDP 0 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 0 --REPEAT 1 --UFOLD_ADD_UNPAIR 0

type_="UFold_ss"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 1 --DIS 0 --CDP 1 --SPTIAL_DIS 0 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0

type_="UFold_DIS"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 0 --DIS 1 --CDP 0 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 0 --REPEAT 0 --UFOLD_ADD_UNPAIR 0

type_="UFold_DIS_ss"
python Evaluate.py --dataset "RSS" --data_path "origin_data" --filename "data/${filename}.lst" --test_filename "data/${test_filename}.lst" \
                --gpu $gpu --metric "F1" \
                "SWIN_UNET" \
                --save_path  "${save_path}/${filename}_SWIN_UNET2_${type_}.pth" \
                --log_path "${log_path}/${filename}_SWIN_UNET2_${type_}_Evaluate.log" \
                --feature_mode 'DNAV' \
                --dataloader_num_workers 5 \
                --batch_size $batch_size \
                --early_stop 100 \
                --epochs 100 \
                --MLDP 1 --DIS 1 --CDP 1 --SPTIAL_DIS 1 --UFOLD 1 --UNPAIR 1 --REPEAT 1 --UFOLD_ADD_UNPAIR 0