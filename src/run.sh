#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

EXPID=exp-gpu1-$(date "+%Y_%m_%d_%H:%M:%S")
PREFIX=full_
# PREFIX=bertsim_

DIR_DATA=$(dirname $(pwd))/data
DIR_PARAGEN=$(dirname $(dirname $(pwd)))/AESOP
DIR_IMOJIE=$(dirname $(dirname $(pwd)))/imojie
DIR_CUR=$(pwd)


# Generate paraphrases
bash generation/generate.sh $DIR_DATA $DIR_PARAGEN

# # Restore knowledge
bash restoration/restore.sh $DIR_DATA $DIR_IMOJIE $DIR_DATA/restore $PREFIX



# Training Model
cd $DIR_IMOJIE
export PYTHONPATH=$DIR_IMOJIE/allennlp
export PATH=$DIR_IMOJIE/allennlp/allennlp:$PATH
python allennlp_script.py \
    --param_path imojie/configs/imojie.json \
    --s models/${PREFIX}${EXPID}AUG \
    --mode train \
    --data ${DIR_DATA}/restore/train/${PREFIX}imojie_aug.tsv \
    --data_val ${DIR_DATA}/dev/carb_sentences.txt \
    --epochs 1 \
    --vocab_path ${DIR_DATA}/vocab/bert
    # --epochs 20


# # Testing Model
cd $DIR_IMOJIE
MODEL_NAME=models/${PREFIX}${EXPID}AUG
# # CaRB
python allennlp_script.py --param_path imojie/configs/imojie.json --s $MODEL_NAME --mode test --test_fp $DIR_OUT/dev/carb
# # CaRB-AutoPara
python allennlp_script.py --param_path imojie/configs/imojie.json --s $MODEL_NAME --mode test --test_fp $DIR_OUT/para/test/paranmt-h2


# Plotting Results
cd $DIR_IMOJIE
python benchmark/pr_plot.py --in=plot_pr/carb --out=plot_pr/carb/pr.png



# Clusters CaRB
cd $DIR_CUR
if [ ! -f cluster/level/results.txt ]; then;
    bash cluster/cluster.sh $DIR_DATA ${DIR_CUR}/cluster
fi

## evaluate on syntactic-specific subsets
MODEL_NAME=$DIR_IMOJIE/models/${PREFIX}${EXPID}AUG
CLUSTER="level"
N_CLS=5

cd $DIR_IMOJIE
for ((i=0; i<=$N_CLS; i ++))
do
    python allennlp_script.py --param_path imojie/configs/imojie.json --s $MODEL_NAME --mode test --test_fp $DIR_DATA/data/dev_test/$CLUSTER"_"$i --perform 'gen_pro_carb'
done
