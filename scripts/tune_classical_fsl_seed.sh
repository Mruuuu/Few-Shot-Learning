###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-04-09 20:59:28
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-04-10 16:17:43
 # @FilePath: /mru/Few-Shot-Learning/scripts/tune_classical_fsl_seed.sh
 # @Description: 
 # 
### 
# change path if needed
LIST=`ls $search_dir`
if echo ${LIST[*]} | grep -qw "main.py"; then
    echo "found"
else
    echo "cd .."
    cd ..
    LIST=`ls $search_dir`
    if echo ${LIST[*]} | grep -qw "main.py"; then
        echo "found"
    else
        echo "not found"
        exit 1
    fi
fi
# run
source myenv/bin/activate
log_root="./logs/tune_classical_fsl_seed"
epoch_size=300
train_mode=classical
num_workers=8
##################
# batch_size=100 #
##################
batch_size=100
tmp=0
for seed in 111 333 555 999
do
    python3 main.py --seed $seed --backbone resnet18 --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.001 --optimizer adamw --num_workers $num_workers --device cuda:2 --fname $((1 + tmp * 3))&
    python3 main.py --seed $seed --backbone resnet18 --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.01 --optimizer sgd --num_workers $num_workers --device cuda:2 --fname $((2 + tmp * 3))&
    python3 main.py --seed $seed --backbone resnet18 --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.001 --optimizer adam --num_workers $num_workers --device cuda:2 --fname $((3 + tmp * 3))&
    wait
    tmp=$((tmp+1))
done