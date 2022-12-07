#!/bin/bash
# Restore knowledge

DIR_DATA=${1:-RobustOIE/data}
DIR_IMOJIE=${2:-RobustOIE/IMOJIE}
DIR_OUT=${3:-RobustOIE/data/restore}
PREFIX=${4:-full_}
# PREFIX={$4:-bertsim_}

export PYTHONPATH=$DIR_IMOJIE/allennlp
export PATH=$DIR_IMOJIE/allennlp/allennlp:$PATH


# Train data
TRAIN_ORI_TSV=$DIR_DATA/train/4cr_qpbo_extractions.tsv
TRAIN_PARA_SRC=$DIR_DATA/para/train/paranmt-h2/4cr_qpbo_sentences-level3.source
TRAIN_PARA_TGT=$DIR_DATA/para/train/paranmt-h2/4cr_qpbo_sentences-level3.tgt
# Dev data
DEV_ORI_TSV=$DIR_DATA/dev/carb/extractions.tsv
DEV_PARA_SRC=$DIR_DATA/para/dev/paranmt-h2/carb_sentences-level3.source
DEV_PARA_TGT=$DIR_DATA/para/dev/paranmt-h2/carb_sentences-level3.tgt
# Test data
TEST_ORI_TSV=$DIR_DATA/test/carb/extractions.tsv
TEST_PARA_SRC=$DIR_DATA/para/test/paranmt-h2/carb_sentences-level3.source
TEST_PARA_TGT=$DIR_DATA/para/test/paranmt-h2/carb_sentences-level3.tgt


# Result
RE_MODEL=./restoration/best_re.pth

if [ ! -d ${DIR_OUT} ]; then
    echo "Creating directory of "${DIR_OUT}
    mkdir -p ${DIR_OUT}/train
    mkdir -p ${DIR_OUT}/dev
    mkdir -p ${DIR_OUT}/test
fi


# --------------------------------------------- Generating Data ------------------------------------
TSVS_ORI=($TRAIN_ORI_TSV $DEV_ORI_TSV $TEST_ORI_TSV)
PARA_SRCS=($TRAIN_PARA_SRC $DEV_PARA_SRC $TEST_PARA_SRC)
PARA_TGTS=($TRAIN_PARA_TGT $DEV_PARA_TGT $TEST_PARA_TGT)
DIRs_OUT=($DIR_OUT/train $DIR_OUT/dev $DIR_OUT/test)
CUDA=0

if [[ $PREFIX =~ full ]]; then BSO=true; else BSO=false; fi;
for((i=0; i<${#TSVS_ORI[@]}; i++)); do
    TGT=${PARA_TGTS[i]##*/}
    if [ ! -f ${DIRs_OUT[i]}/${TGT: -0: -4}.restore.${PREFIX}final.tsv ]; then
        python restoration/map_from_tree.py  \
            --triples-tsv ${TSVS_ORI[i]} \
            --para-source ${PARA_SRCS[i]} \
            --para-tgt ${PARA_TGTS[i]} \
            --out-folder ${DIRs_OUT[i]} \
            --out-file-prefix ${PREFIX} \
            --cuda-device $CUDA \
            --model-path ${RE_MODEL} \
            --bert-sim-only $BSO
    fi
done


python repair.py \
    --in_file ${DIR_OUT}/${PREFIX}result.tsv \
    --out_file ${DIR_OUT}/${PREFIX}result.tsv \
    --n_cpus 40 \
    --train_pre_files $DIR_DATA/train/4cr_qpbo_extractions.tsv \
    --best_model_path restoration/save_models



# --------------------------------------------- Merge Data ------------------------------------
TGT=${TRAIN_PARA_TGT##*/}
cat ${DIR_OUT}/train/${TGT: -0: -4}.restore.${PREFIX}final.tsv $DIR_DATA/train/4cr_qpbo_extractions.tsv > ${DIR_OUT}/train/${PREFIX}imojie_aug.tsv
cat ${DIR_OUT}/train/${TGT: -0: -4}.restore.${PREFIX}final.tsv | awk -F '\t' '{print $1}' > ${DIR_OUT}/train/${PREFIX}sentences.txt

