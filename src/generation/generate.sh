#!/bin/bash
# Generate paraphrase data

DIR_DATA=${1:-RobustOIE/data}
DIR_PARAGEN=${2:-AESOP}

export PATH="${DIR_PARAGEN}:$PATH"


# select ** pre-trained models ** and the ** statistic files ** for paraphrase generation
if true
then
    PARA_MODEL=paranmt-h2
    PARA_STAT=ParaNMT-hf-refine
else
    PARA_MODEL=qqppos-h2
    PARA_STAT=QQPPos-hf-refine
fi

# --------------------------------------------- Select Data ------------------------------------

# Train data
FILE_ORI_TRAIN=$DIR_DATA/train/4cr_qpbo_extractions.tsv
FILE_IN_TRAIN=$DIR_DATA/para/train/4cr_qpbo_sentences.txt
FILE_OUT_TRAIN=$DIR_DATA/para/train/${PARA_MODEL}/4cr_qpbo_sentences.txt

# Dev data
FILE_ORI_DEV=$DIR_DATA/dev/carb/extractions.tsv
FILE_IN_DEV=$DIR_DATA/para/dev/carb_sentences.txt
FILE_OUT_DEV=$DIR_DATA/para/dev/${PARA_MODEL}/carb_sentences.txt

# Test data
FILE_ORI_TEST=$DIR_DATA/test/carb/extractions.tsv
FILE_IN_TEST=$DIR_DATA/para/test/carb_sentences.txt
FILE_OUT_TEST=$DIR_DATA/para/test/${PARA_MODEL}/carb_sentences.txt

DIRS_OUT=($(dirname ${FILE_OUT_TRAIN}) $(dirname ${FILE_OUT_DEV}) $(dirname ${FILE_OUT_TEST}))
for OUT in ${DIRS_OUT[@]}; do
    if [ ! -d ${OUT} ]; then
        echo "Creating directory of "${OUT}
        mkdir -p ${OUT}
    fi
done

# --------------------------------------------- Prepare Data ------------------------------------

# Generate original sentences
FILES_ORI=($FILE_ORI_TRAIN $FILE_ORI_DEV $FILE_ORI_TEST)
FILES_IN=($FILE_IN_TRAIN $FILE_IN_DEV $FILE_IN_TEST)

for((i=0; i<${#FILES_ORI[@]}; i++)); do
    ORI=${FILES_ORI[i]}
    IN=${FILES_IN[i]}
    if [ ! -f ${IN} ]; then
    #     # according to the format of input file of paraphrase-generating model, each 2 lines serve as a sample (input, gold)
    #     cat ${ORI} | awk -F '\t' 'BEGIN{d[0]=xx} {if ($1!=d[NR-1]) {print $1; print $1}; d[NR]=$1}' > ${IN}
        cat ${ORI} | awk -F '\t' 'BEGIN{d[0]=xx} {if ($1!=d[NR-1]) {print $1}; d[NR]=$1}' > ${IN}
    fi
done



# --------------------------------------------- Generate Paraphrases ------------------------------------

DIR_CUR=$(pwd)
cd ${DIR_PARAGEN}
LEVEL=3

FILES_IN=($FILE_IN_TRAIN $FILE_IN_DEV $FILE_IN_TEST)
FILES_OUT=(${FILE_OUT_TRAIN: -0: -4}-level${LEVEL}.extract ${FILE_OUT_DEV: -0: -4}-level${LEVEL}.extract ${FILE_OUT_TEST: -0: -4}-level${LEVEL}.extract)

for((i=0; i<${#FILES_IN[@]}; i++)); do
    if [ ! -f ${FILES_OUT[i]} ]; then
        IN=${FILES_IN[i]}
        OUTDIR=${DIRS_OUT[i]}
        # nohup python run_predict.py \
        python -u run_predict.py \
            --model_name pretrained-models/${PARA_MODEL} \
            --input_file ${IN} \
            --output_dir ${OUTDIR} \
            --statistics_path processed-data/qiji/${PARA_STAT}/repe_statistics \
            --level $LEVEL \
            --n_gpus 1 \
            --n_cpus 40 \
            --batch_size 32
        #     > ${DIR_CUR}/aesop_gen_para.log &

        #     --model_name pretrained-models/qqppos-h2 \
        #     --fp16 \
    fi
done

# clean caches
ls -l | awk '{if ($9 ~ /^corenlp_server-.*?props$/) print "rm "$9 | "/bin/bash"}'

