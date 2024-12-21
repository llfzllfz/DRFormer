log_path="./log/ArchiveII"
save_path="./models/ArchiveII"
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

filename="ArchiveII_train_112"
test_filename="ArchiveII_valid_112"

batch_size=16
epochs=100
early_stop=100
gpu=1


