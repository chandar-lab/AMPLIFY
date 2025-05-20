
python -u main.py \
        experiments=ESM2_TEST \
        experiments.prt_model_name='ism' \
        experiments.ft_model_path='None' \
        experiments.device=0

python -u main.py \
        experiments=ESM2_TEST \
        experiments.prt_model_name='esm-s' \
        experiments.ft_model_path='None' \
        experiments.device=0
