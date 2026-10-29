#!/bin/bash

# compete for GPU source:
export compete=1
if [ $compete ]; then
  while true; do

    # 提取gpu占用比例
    idle_time=$(gpustat | grep 'RTX' | awk '{print $11}')

    echo -e "\n\n"
    date

      # 检查是否gpu占用比例低于30%，用exit命令退出循环
      if [ $idle_time -lt 2457 ]; then
      echo -e "\n"
          echo "满足条件，退出循环"
          break
      fi

      echo -e "\n\t\t failed, keep trying... \n\n"
      sleep 5m  # 休眠5分钟
  done
fi



export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/home/lenovo/mmdetection-ylzh-2025

# 参数选择: 实验config编号; 测试的epoch;
listExps=(
#    mm25_hrrsd_faster-rcnn-o2xy_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o4xyxw_r50_fpn_1x
#    sclgt0_faster-rcnn_r50_fpn_1x_hrrsd
#    dior_faster-rcnn_r50_fpn_lr02_2x_dior
#    mm25_hrrsd_faster-rcnn-o2xyxwvar-dvd8sqrt_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxwvar-mul2sqrt_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw-Wb10_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw-Wb100_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw-WbE_1_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw-Wb5E_1_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw-Wb3_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn-o2xyxw-Wb3E_1_r50_fpn_1x

#    mm25_hrrsd_faster-rcnn_r50_fpn_1x_tiny10
#    mm25_hrrsd_faster-rcnn_r50_head-202121-base_fpn_1x

#    mm25_hrrsd_faster-rcnn_DFR_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_DFR2_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_DFR2_o2xyxw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-l1_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-l1-10_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-l1-30_r50_fpn_1x

#    mm25_hrrsd_faster-rcnn_DFR_soar-l1-10_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-l1-10_o2xyxw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_DFR_soar-l1-10_o2xyxw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_DFR_soar-l1-3_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-l1-3_o2xyxw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_DFR_soar-l1-3_o2xyxw_r50_fpn_1x 

    mm25_hrrsd_faster-rcnn_DFR_soar-l1-1_o2xyxw_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-sl1_r50_fpn_1x
#    mm25_hrrsd_faster-rcnn_soar-cE_r50_fpn_1x
)

export e=12 # epochs
export bc=6
export nw=6
export root=projects/DSS/configs

# -------------------- non-repeat exps --------------------

for i in ${listExps[@]}
	do
    # 训练：
    #    逻辑：
    #        检查checkpoint文件,如果找到就检查有没有相应的pth,有则resume
    if [ -f "work_dirs/${i}/last_checkpoint" ] && [ -f $(cat work_dirs/${i}/last_checkpoint) ]; then
        echo -e "\n\n\n\n \t\tStart training with resume checkpoint from work_dirs/${i}/last_checkpoint \n\n\n\n"
        python tools/train.py ${root}/${i}.py --auto-scale-lr \
            --cfg-options train_dataloader.batch_size=${bc} \
            --cfg-options val_dataloader.batch_size=${bc} \
            --cfg-options train_dataloader.num_workers=${nw} \
            --cfg-options val_dataloader.num_workers=${nw} --resume
    else
        echo -e '\n\n\n\n \t\tStart training without resume \n\n\n\n'
        python tools/train.py ${root}/${i}.py --auto-scale-lr \
            --cfg-options train_dataloader.batch_size=${bc} \
            --cfg-options val_dataloader.batch_size=${bc} \
            --cfg-options train_dataloader.num_workers=${nw} \
            --cfg-options val_dataloader.num_workers=${nw} #--resume
    fi

    # 预先留存：
    cp work_dirs/${i}/epoch_${e}.pth \
	    work_dirs/${i}/epoch_${e}_save.pth

    # '''
    # 清盘：
    #    逻辑：
    #        如果有 \*_save.pth, 表示训练完了，就可以清盘了, 否则就不清盘
    if [ -f "work_dirs/${i}/epoch_${e}_save.pth" ]; then
        if [ $e -eq 24 ]; then
            rm work_dirs/${i}/epoch_?.pth
            rm work_dirs/${i}/epoch_1?.pth
            rm work_dirs/${i}/epoch_2[0,1,2,3].pth
            echo "delete for 24"
        elif [ $e -eq 12 ]; then
            rm work_dirs/${i}/epoch_?.pth
            rm work_dirs/${i}/epoch_1?.pth
            echo "delete for 12"
        else
            export ep=$((e/10-1))
            # echo "ep=${ep}"
            if [ $ep -eq 0 ]; then
                rm work_dirs/${i}/epoch_?.pth
                echo "delete for 1-9"
            else
                for (( n=${ep}; n>1; n-- )); do
                    rm work_dirs/${i}/epoch_${n}?.pth
                    echo "repeat delete *1-*9"
                done
            fi
        fi
    else
        echo "no epoch_e_save."
    fi
  # '''

  # 测试：

#	   echo -e '\n\n \t\tStart testing ... \n\n'
#    python tools/test.py \
#	    ${root}/${i}.py \
#	    work_dirs/${i}/epoch_${e}_save.pth \
#	    --out work_dirs/${i}/results_save_000.pkl \
#	    --cfg-options test_dataloader.batch_size=6 \
#	    --cfg-options test_dataloader.num_workers=6 \
#	    | tee -a work_dirs/${i}/test_ylzh_all.log


  # --------------------------------------------------------------------------------------------------------------------

  # classwise test
  # ${root}/${i}.py \
	  echo -e '\n\n \t\tStart classwise test ... \n\n'
#    python tools/test.py \
#	    work_dirs/${i}/${i}.py \
    python tools/test.py \
	    ${root}/${i}.py \
	    work_dirs/${i}/epoch_${e}_save.pth \
	    --out work_dirs/${i}/results_save.pkl \
	    --cfg-options test_dataloader.batch_size=6 \
	    --cfg-options test_dataloader.num_workers=6 \
	    --cfg-options test_evaluator.classwise=True \
	    | tee -a work_dirs/${i}/test_ylzh_all.log

#    # 使用tail和head获取倒数5-16行,然后使用awk处理每行
#    filename=work_dirs/${i}/test_ylzh_all.log
#    tail -n 15 "$filename" | head -n 12 | awk '{
#        for (j=NF; j>0; j--) {
#            if ($j ~ /[0-9.]+/) {
#
#                # 分割字符串并取=后面的部分
#                suffix = substr($j, index($j, "=") + 1);
#
#                # 去除非数字字符（如果需要保留小数点，则不要替换小数点）
#                gsub(/[^0-9.]/, "", $j);
#
#                print 100*$j;
#                break; # 找到第一个=后就跳出循环
#            }
#        }
#    }'
done