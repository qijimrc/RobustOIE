#!/bin/bash
# Clustering CaRB into syntactic-specific subsets based on the HW distance
# Compute average distance between the subsets and OpenIE4 dataset
# Evaluate IMOJIE on each syntactic-specific subset

DIR_DATA=${1:-RobustOIE/data}
DIR_RUN=${2:-.}
# if [ ! $DIR_RUN ]; then DIR_RUN=$(pwd); fi;

# Target Dir to evaluate
TARGET_DIR=$DIR_DATA/data/dev_test
# Train source
TRAIN_SORCE=$DIR_DATA/para/train/paranmt-h2/4cr_qpbo_sentences-level3.source
# CaRB
CaRB_DEV_SOURCES=$DIR_DATA/para/dev/paranmt-h2/carb_sentences-level3.source
CaRB_TEST_SOURCES=$DIR_DATA/para/test/paranmt-h2/carb_sentences-level3.source
CaRB_DEV_EXT=$DIR_DATA/dev/carb/extractions.tsv
CaRB_TEST_EXT=$DIR_DATA/test/carb/extractions.tsv


# --------------------------------------------- Clustering ------------------------------------
python $DIR_RUN/k_means.py \
    --carb_dev_src $CaRB_DEV_SOURCES \
    --carb_test_src $CaRB_TEST_SOURCES \
    --train_source $TRAIN_SORCE \
    --carb_dev_ext $CaRB_DEV_EXT \
    --carb_dev_ext $CaRB_TEST_EXT \
    --target_dir $TARGET_DIR \
    --tag_dict $DIR_RUN/tag_dict.json


