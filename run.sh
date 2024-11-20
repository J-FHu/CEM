export CUDA_VISIBLE_DEVICES=0
# SR
python demo_causal_effect_calculation.py --config demo-SR_CEM.yml
python demo_CEM_generation.py --config demo-SR_CEM.yml
# DR
python demo_causal_effect_calculation.py --config demo-DR_CEM.yml
python demo_CEM_generation.py --config demo-DR_CEM.yml
# DN
python demo_causal_effect_calculation.py --config demo-DN_CEM.yml
python demo_CEM_generation.py --config demo-DN_CEM.yml

