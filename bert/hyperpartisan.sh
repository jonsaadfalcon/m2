
cd ../code
data_dir=/future/u/jonsf/HazyLab/m2/bert/datasets/
dataset=hyperpartisan

seed=52
length=512
output=${dataset}_length_${length}_seed_${seed}
output_dir=${data_dir}/TEMP/0330A_1_${output}_$(date +%F-%H-%M-%S-%N)
output_metrics_filepath = ${data_dir}${dataset}

if ! test -f "/future/u/jonsf/HazyLab/m2/bert/datasets/hyperpartisan/${output}.json"; then
  python3 train.py \
  --task_name multilabel \
  --dataset_name $dataset \
  --output_metrics_filepath ../results/1/train/${output}.json \
  --model_dir $data_dir/Corpora/RoBERTa/mimic_roberta_base \
  --seed $seed \
  --train_filepath ../data/sample.json \
  --dev_filepath ../data/sample.json \
  --output_dir $output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --metric_for_best_model micro_f1 \
  --greater_is_better \
  --max_seq_length $length \
  --segment_length 64 --do_use_stride --do_use_label_wise_attention

  python3 eval.py \
  --task_name multilabel \
  --dataset_name $dataset \
  --output_metrics_filepath "/future/u/jonsf/HazyLab/m2/bert/datasets/hyperpartisan/${output}.json" \
  --model_dir $output_dir \
  --test_filepath ../data/sample.json \
  --output_dir $output_dir \
  --max_seq_length $length \
  --segment_length 64 --do_use_stride --do_use_label_wise_attention

  rm -r $output_dir
fi