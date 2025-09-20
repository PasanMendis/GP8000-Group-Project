# GP8000-Group-Project
This is a group project for the module GP8000


Guide for the codebase

1. Install the requirements
2. Generate metadata text prompt using build_prompts.py. It will save that text prompt to the given file in another column.

    python3 build_prompts.py df_filtered_year.csv

3. Run data_split.py to append new data to train, test and val files.

    python3 data_split.py Metadata_Generate/df_filtered_year.csv

4. python3 train_stageA.py --train train.csv --val val.csv --gpu 1 --model openai/clip-vit-base-patch32

5. python3 train_stageB.py \
  --train train.csv --val val.csv \
  --gpu 1 --model openai/clip-vit-base-patch32 \
  --text_cols meta_prompt,overview,logline \
  --epochs 12 --batch 64 --lr_head 1e-3 --lr_enc 5e-6