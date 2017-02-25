#!/bin/sh

python visualize_mechturk_statistics.py 70 'more'
python visualize_mechturk_statistics.py 80 'more'
python visualize_mechturk_statistics.py 70 'less'
python visualize_mechturk_statistics.py 80 'less'
python visualize_mechturk_statistics_alg_vs_human.py 70
python visualize_mechturk_statistics_alg_vs_human.py 80
