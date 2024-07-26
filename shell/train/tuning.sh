#!/bin/bash
# peft training shell code

type="" # tune_mode 
jsons=() # *-fold
targets=() # peft 
rf="" # ADPT - reduction factor
r="" # LoRA - rank
alpha=$((2*$r)) # LoRA - alpha 
num="" # number of train data

for json in "${jsons[@]}"; do
    for target in "${targets[@]}"; do
        if [ "$target" -eq 136 ]; then 
            resolution="192192136"
            space_x="1.21875"
            space_y="1.21875"
            space_z="1.21875"
            roi_z="64"
        elif [ "$target" -eq 192 ]; then
            resolution="192192128"
            space_x="1.21875"
            space_y="1.21875"
            space_z="1.21875"
            roi_z="64"
        elif [ "$target" -eq 224 ]; then
            resolution="22422481"
            space_x="1.01821"
            space_y="1.01821"
            space_z="2.0269988"
            roi_z="64"
        elif [ "$target" -eq 90 ]; then
            resolution="12812890"
            space_x="2"
            space_y="2"
            space_z="2"
            roi_z="64"
        else
            resolution="12812863"
            space_x="2.05941"
            space_y="2.05941"
            space_z="2.4250016"
            roi_z="32"
        fi

        for folder in 136 192 224 90 63; do
                if [ "$folder" -eq "$target" ]; then
                    continue
                fi

                folder_dir="${folder}_${target}#${json}" 
                
            python3 -u /main.py \
                    --data_dir " /root_dir/ADNI/Dynamic/Resolution/${resolution}/" \
                    --json_list " /root_dir/ADNI/Dynamic/Resolution/${resolution}/${resolution}_${json}_${num}.json" \
                    --logdir "" \
                    --pretrained_dir "./base_raw/Resolution/${folder}#${json}/" \
                    --pretrained_model_name "./base_raw/Resolution/final_${folder}#${json}.pt" \
                    --csv_dir "/root_dir/csv/${type}/num_${num}.xlsx"\
                    --optim_lr "" \
                    --reg_weight "" \
                    --optim_name "" \
                    --lrschedule "" \
                    --space_x "$space_x" \
                    --space_y "$space_y" \
                    --space_z "$space_z" \
                    --roi_z "$roi_z" \
                    --en1_tokens "0" \
                    --en2_tokens "8" \
                    --en3_tokens "32" \
                    --rf "$rf" \
                    --r "$r" \
                    --lora_alpha "$alpha" \
                    --tuning \
                    --tune_mode "$type" \
                    --filename "${folder_dir}" \
                    --wandb_project "Masterthesis_cvt_${type}_${num}" \
                    --save_checkpoint 
        done
    done
done