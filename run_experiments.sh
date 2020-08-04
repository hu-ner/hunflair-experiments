#!/usr/bin/env bash

echo "Predict - PDR - HunFlair"
python predict_hunflair.py --input_file corpora/craft_v4.tsv --output_file evaluation/hunflair/craft.ann

echo "Predict - PDR - HunFlair"
python predict_hunflair.py --input_file corpora/bionlp2013cg.tsv --output_file evaluation/hunflair/bionlp2013cg.ann

echo "Predict - PDR - HunFlair"
python predict_hunflair.py --input_file corpora/pdr.tsv --output_file evaluation/hunflair/pdr.ann


# Prediction spacy

echo "Predict - CRAFT - SciSpacy (en_ner_bionlp13cg_md)"
python predict_scispacy.py --input corpora/craft_v4.tsv --model en_ner_bionlp13cg_md --output_file evaluation/spacy_en_ner_bionlp13cg_md/craft.ann
echo "Predict - CRAFT - SciSpacy (en_ner_jnlpba_md)"
python predict_scispacy.py --input corpora/craft_v4.tsv --model en_ner_jnlpba_md --output_file evaluation/spacy_en_ner_jnlpba_md/craft.ann
echo "Predict - CRAFT - SciSpacy (en_ner_bc5cdr_md)"
python predict_scispacy.py --input corpora/craft_v4.tsv --model en_ner_bc5cdr_md --output_file evaluation/spacy_en_ner_bc5cdr_md/craft.ann

## -----------
echo "Predict - BioNLP13CG - SciSpacy (en_ner_craft_md)"
python predict_scispacy.py --input corpora/bionlp2013cg.tsv --model en_ner_craft_md --output_file evaluation/spacy_en_ner_craft_md/bionlp2013cg.ann
echo "Predict - BioNLP13CG - SciSpacy (en_ner_jnlpba_md)"
python predict_scispacy.py --input corpora/bionlp2013cg.tsv --model en_ner_jnlpba_md --output_file evaluation/spacy_en_ner_jnlpba_md/bionlp2013cg.ann
echo "Predict - BioNLP13CG - SciSpacy (en_ner_bc5cdr_md)"
python predict_scispacy.py --input corpora/bionlp2013cg.tsv --model en_ner_bc5cdr_md --output_file evaluation/spacy_en_ner_bc5cdr_md/bionlp2013cg.ann

## -----------

echo "Predict - PDR - SciSpacy (en_ner_bionlp13cg_md)"
python predict_scispacy.py --input corpora/pdr.tsv --model en_ner_bionlp13cg_md --output_file evaluation/spacy_en_ner_bionlp13cg_md/pdr.ann
echo "Predict - PDR - SciSpacy (en_ner_craft_md)"
python predict_scispacy.py --input corpora/pdr.tsv --model en_ner_craft_md --output_file evaluation/spacy_en_ner_craft_md/pdr.ann
echo "Predict - PDR - SciSpacy (en_ner_jnlpba_md)"
python predict_scispacy.py --input corpora/pdr.tsv --model en_ner_jnlpba_md --output_file evaluation/spacy_en_ner_jnlpba_md/pdr.ann
echo "Predict - PDR - SciSpacy (en_ner_bc5cdr_md)"
python predict_scispacy.py --input corpora/pdr.tsv --model en_ner_bc5cdr_md --output_file evaluation/spacy_en_ner_bc5cdr_md/pdr.ann

#########################################################################################################################

echo
echo
echo "            CRAFT - Chemical                                "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/misc/misc_craft.ann 2>&1 | grep chemical | tail -n2
echo
echo "SciSpacy (en_ner_bionlp13cg_md)"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/spacy_en_ner_bionlp13cg_md/craft.ann 2>&1 | grep chemical | tail -n2
echo
echo "SciSpacy (en_ner_bc5cdr_md)"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/spacy_en_ner_bc5cdr_md/craft.ann 2>&1 | grep chemical | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/hunflair/craft.ann 2>&1 | grep chemical | tail -n2
echo

## ------------------------

echo
echo
echo "            CRAFT - Gene                                "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/misc/misc_craft.ann 2>&1 | grep gene | tail -n2
echo
echo "SciSpacy (en_ner_bionlp13cg_md)"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/spacy_en_ner_bionlp13cg_md/craft.ann 2>&1 | grep gene | tail -n2
echo
echo "SciSpacy (en_ner_jnlpba_md)"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/spacy_en_ner_jnlpba_md/craft.ann 2>&1 | grep gene | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/hunflair/craft.ann 2>&1 | grep gene | tail -n2
echo

## ------------------------

echo
echo "            CRAFT - Species                                "
echo -----------------------------------------------------------
echo
echo "MISC"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/misc/misc_craft.ann 2>&1 | grep species | tail -n2
echo
echo "SciSpacy (en_ner_bionlp13cg_md)"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/spacy_en_ner_bionlp13cg_md/craft.ann 2>&1 | grep species | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/craft_v4.tsv --gold_file corpora/craft_v4.ann --pred_file evaluation/hunflair/craft.ann 2>&1 | grep species | tail -n2
echo

###################################################################################################################################################################

echo
echo
echo "                   BioNLP13 CG Chemical                       "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/misc/misc_bionlp.ann 2>&1 | grep chemical | tail -n2
echo
echo "SciSpacy (en_ner_craft_md)"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/spacy_en_ner_craft_md/bionlp2013cg.ann 2>&1 | grep chemical | tail -n2
echo
echo "SciSpacy (en_ner_bc5cdr_md)"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/spacy_en_ner_bc5cdr_md/bionlp2013cg.ann 2>&1 | grep chemical | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/hunflair/bionlp2013cg.ann 2>&1 | grep chemical | tail -n2
echo

## ------------------------

echo
echo "                   BioNLP13 CG Disease                       "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/misc/misc_bionlp.ann 2>&1 | grep disease | tail -n2
echo
echo "SciSpacy (en_ner_bc5cdr_md)"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/spacy_en_ner_bc5cdr_md/bionlp2013cg.ann 2>&1 | grep disease | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/hunflair/bionlp2013cg.ann 2>&1 | grep disease | tail -n2
echo

## ------------------------

echo
echo "                   BioNLP13 CG Gene                       "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/misc/misc_bionlp.ann 2>&1 | grep gene | tail -n2
echo
echo "SciSpacy (en_ner_craft_md)"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/spacy_en_ner_craft_md/bionlp2013cg.ann 2>&1 | grep gene | tail -n2
echo
echo "SciSpacy (en_ner_jnlpba_md)"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/spacy_en_ner_jnlpba_md/bionlp2013cg.ann 2>&1 | grep gene | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/hunflair/bionlp2013cg.ann 2>&1 | grep gene | tail -n2
echo

## ------------------------

echo
echo "                   BioNLP13 CG Species                       "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/misc/misc_bionlp.ann 2>&1 | grep species | tail -n2
echo
echo "SciSpacy (en_ner_craft_md)"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/spacy_en_ner_craft_md/bionlp2013cg.ann 2>&1 | grep species | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/bionlp2013cg.tsv --gold_file corpora/bionlp2013cg.ann --pred_file evaluation/hunflair/bionlp2013cg.ann 2>&1 | grep species | tail -n2
echo

###################################################################################################################################################################

echo
echo
echo "                   Plant Disease (PDR) - Disease          "
echo -----------------------------------------------------------
echo "MISC"
python evaluate_ann.py --text_file corpora/pdr.tsv --gold_file corpora/pdr.ann --pred_file evaluation/misc/misc_pdr.ann 2>&1 | grep disease | tail -n2
echo
echo "SciSpacy (en_ner_bionlp13cg_md)"
python evaluate_ann.py --text_file corpora/pdr.tsv --gold_file corpora/pdr.ann --pred_file evaluation/spacy_en_ner_bionlp13cg_md/pdr.ann 2>&1 | grep disease | tail -n2
echo
echo "SciSpacy (en_ner_bc5cdr_md)"
python evaluate_ann.py --text_file corpora/pdr.tsv --gold_file corpora/pdr.ann --pred_file evaluation/spacy_en_ner_bc5cdr_md/pdr.ann 2>&1 | grep disease | tail -n2
echo
echo "HunFlair"
python evaluate_ann.py --text_file corpora/pdr.tsv --gold_file corpora/pdr.ann --pred_file evaluation/hunflair/pdr.ann 2>&1 | grep disease | tail -n2
