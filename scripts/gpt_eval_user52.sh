#!/bin/bash

user_id=52
hsz=1020
checkpoint=4
bsz=64
stride=3
decode="softmax"
temp=0.5
num_seed_trans=2

cd ..

data_dir="./data/alfa/"
output_dir="./output_bert/"

python gpt_eval.py \
  --output_dir "${output_dir}" \
  --data_dir "${data_dir}" \
  --data_fname "small_transaction"\
  --user_id "${user_id}" \
  --checkpoint "${checkpoint}" \
  --hidden_size "${hsz}" \
  --batch_size "${bsz}" \
  --stride "${stride}" \
  --decoding "${decode}" \
  --temperature "${temp}" \
  --num_seed_trans "${num_seed_trans}" \
  --data_extension "" \
  --store_csv