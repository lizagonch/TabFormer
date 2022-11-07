#!/bin/bash

user_id=52
data_root="./data/alfa/"
checkpoint="./checkpoints/credit_card/gpt2-userid_${user_id}-nbins_10-hsz_${user_id}"

cd ..
python main.py \
                --lm_type gpt2 \
                --field_ce \
                --flatten \
                --data_root  "${data_root}"\
                --data_fname "transaction_0"\
                --data_extension "userid-${user_id}" \
                --user_ids "${user_id}" \
                --output_dir "${checkpoint}" \
                --checkpoint 0 \
                --do_train \
                --save_steps 1 \
                --num_train_epochs 2 \
                --stride 3 \
                --field_hs 1020 \
                --cached