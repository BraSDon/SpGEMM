#!/bin/bash

set -e

DATA_DIR="./data"
mkdir -p $DATA_DIR

NEMETH_URL="https://suitesparse-collection-website.herokuapp.com/MM/Nemeth/nemeth07.tar.gz"
LHR71C_URL="https://suitesparse-collection-website.herokuapp.com/MM/Mallya/lhr71c.tar.gz"
C71_URL="https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/c-71.tar.gz"
PREF_URL="https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/preferentialAttachment.tar.gz"
CONSPH_URL="https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz"
RGG_URL="https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_22_s0.tar.gz"
ASIC_URL="https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_680ks.tar.gz"
RAJAT_URL="https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat31.tar.gz"
M6_URL="https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/M6.tar.gz"

URLS=($NEMETH_URL $LHR71C_URL $C71_URL $PREF_URL $CONSPH_URL $RGG_URL $ASIC_URL $RAJAT_URL $M6_URL)
for URL in ${URLS[@]}; do
    echo "Downloading $URL"
    wget -P $DATA_DIR $URL
    tar -xzf $DATA_DIR/$(basename $URL) -C $DATA_DIR
    rm $DATA_DIR/$(basename $URL)
done
