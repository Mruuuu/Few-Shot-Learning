###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-03-14 10:52:31
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-04-01 01:38:03
 # @FilePath: /mru/Few-Shot-Learning/scripts/tune_oprimizer_lr.sh
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
log_root="./logs/tune_optimizer_lr"
epoch_size=300
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.01 --optimizer adam --num_workers 4 --device cuda:3 --fname 1&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.001 --optimizer adam --num_workers 4 --device cuda:3 --fname 2&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.0001 --optimizer adam --num_workers 4 --device cuda:3 --fname 3&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.00001 --optimizer adam --num_workers 4 --device cuda:3 --fname 4&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.01 --optimizer rmsprop --num_workers 4 --device cuda:3 --fname 5&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.001 --optimizer rmsprop --num_workers 4 --device cuda:3 --fname 6&
wait
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.0001 --optimizer rmsprop --num_workers 4 --device cuda:3 --fname 7&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.00001 --optimizer rmsprop --num_workers 4 --device cuda:3 --fname 8&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.01 --optimizer sgd --num_workers 4 --device cuda:3 --fname 9&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.001 --optimizer sgd --num_workers 4 --device cuda:3 --fname 10&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.0001 --optimizer sgd --num_workers 4 --device cuda:3 --fname 11&
python3 main.py --log_root $log_root --epoch_size $epoch_size --lr 0.00001 --optimizer sgd --num_workers 4 --device cuda:3 --fname 12&
wait