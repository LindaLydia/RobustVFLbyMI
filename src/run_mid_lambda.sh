# # No Active Party Top Model
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4
# python vfl_main_task_no_defense.py --apply_encoder True --epochs 30
# python vfl_dlg_no_defense.py --apply_encoder True
# python vfl_main_task_no_defense.py --epochs 30
# python vfl_dlg_no_defense.py

# # With Active Party Top Model
# # MID
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --apply_trainable_layer True
# # Confusional Auto Encoder
# python vfl_main_task_no_defense.py --apply_encoder True --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_encoder True --apply_trainable_layer True
# No Derense
python vfl_main_task_no_defense.py --epochs 30 --apply_trainable_layer True
python vfl_dlg_no_defense.py --apply_trainable_layer True