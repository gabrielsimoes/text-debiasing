# Text Debiasing

A project for a class on Natural Language Processing (CSE 40657). A model for debiasing a text. Instructions for running:

- Data must be downloaded and saved to `bias_data/` from https://github.com/rpryzant/neutralizing-bias.

- Model can be trained with `python train.py --model transformer --train bias_data/WNC/biased.word.train --dev bias_data/WNC/biased.word.dev`. Checkpoints will be automatically saved to `transformer-$i.model`.

- Model output can be generated with `python train.py --model transformer --test bias_data/WNC/biased.word.test --load transformer-$i.model > test.model`

- Gold test sentences have to be saved for comparisson with `cat bias_data/WNC/biased.word.test | cut -d$'\t' -f  3 > test.gold`.

- Output can be scored with `python score.py test.gold test.model`.
