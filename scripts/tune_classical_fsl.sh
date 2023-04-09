###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-03-14 10:52:31
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-04-01 22:13:50
 # @FilePath: /mru/Few-Shot-Learning/scripts/tune_classical_fsl.sh
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
tmp=0
log_root="./logs/tune_classical_fsl"
epoch_size=300
train_mode=classical
num_workers=8
##################
# batch_size=100 #
##################
batch_size=100
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.01 --optimizer adam --num_workers $num_workers --device cuda:2 --fname 1&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.001 --optimizer adam --num_workers $num_workers --device cuda:2 --fname 2&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.0001 --optimizer adam --num_workers $num_workers --device cuda:2 --fname 3&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.00001 --optimizer adam --num_workers $num_workers --device cuda:2 --fname 4&
wait
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.01 --optimizer adamw --num_workers $num_workers --device cuda:2 --fname 5&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.001 --optimizer adamw --num_workers $num_workers --device cuda:2 --fname 6&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.0001 --optimizer adamw --num_workers $num_workers --device cuda:2 --fname 7&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.00001 --optimizer adamw --num_workers $num_workers --device cuda:2 --fname 8&
wait
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.01 --optimizer sgd --num_workers $num_workers --device cuda:2 --fname 9&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.001 --optimizer sgd --num_workers $num_workers --device cuda:2 --fname 10&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.0001 --optimizer sgd --num_workers $num_workers --device cuda:2 --fname 11&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.00001 --optimizer sgd --num_workers $num_workers --device cuda:2 --fname 12&
wait
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.01 --optimizer adadelta --num_workers $num_workers --device cuda:2 --fname 13&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.001 --optimizer adadelta --num_workers $num_workers --device cuda:2 --fname 14&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.0001 --optimizer adadelta --num_workers $num_workers --device cuda:2 --fname 15&
python3 main.py --log_root $log_root --train_mode $train_mode --batch_size $batch_size --epoch_size $epoch_size --lr 0.00001 --optimizer adadelta --num_workers $num_workers --device cuda:2 --fname 16&
wait
