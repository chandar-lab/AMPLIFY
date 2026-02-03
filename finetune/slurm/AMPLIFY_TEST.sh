ft_model_paths=("None" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_aido_gearnet_1_1_0.5_0.5_loss_large_0.8_aido_gearnet_1" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_af2_1_1_0.5_0.5_loss_large_0.8_foldseek_af2_1" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.0_0.0_loss_large_0.8_foldseek_gearnet_1" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.0_0.5_loss_large_0.8_foldseek_gearnet_1" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.5_0.0_loss_large_0.8_foldseek_gearnet_1" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.5_0.5_loss_large_0.8_foldseek_gearnet_1" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.5_0.5_loss_large_0.8_foldseek_gearnet_2" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.5_0.5_loss_large_0.8_foldseek_gearnet_3" "AMPLIFY_350M_AMPLIFY_120M_None_1_0.5_0.5_loss_large_1.0_protoken_gearnet_1_1_0.5_0.5_loss_large_0.8_protoken_gearnet_1" "AMPLIFY_350M_None_1_0.5_0.5_loss_large_0.8_foldseek_gearnet_1" "AMPLIFY_350M_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1" "AMPLIFY_350M_None_1_0.5_0.5_loss_small_0.8_foldseek_gearnet_1")

for ft_model_path in "${ft_model_paths[@]}"; do
    python -u main.py \
        experiments=AMPLIFY_TEST \
        experiments.ft_model_path="$ft_model_path"\
        experiments.device=0
done
