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


# With Active Party Top Model
python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 300 --apply_trainable_layer True --batch_size 1
python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --apply_trainable_layer True --gpu 1
python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 300 --apply_trainable_layer True --batch_size 1
python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --apply_trainable_layer True --gpu 1
python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 300 --apply_trainable_layer True --batch_size 1
python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --apply_trainable_layer True --gpu 1
python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 300 --apply_trainable_layer True --batch_size 1
python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --apply_trainable_layer True --gpu 1
python vfl_main_task_no_defense.py --apply_encoder True --epochs 300 --apply_trainable_layer True --batch_size 1
python vfl_dlg_no_defense.py --apply_encoder True --apply_trainable_layer True --gpu 1