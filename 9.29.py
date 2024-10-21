python main.py --dir_data /mnt/diskb/penglong/zzff/Pretrained-IPT-main/data/dataset --pretrain /mnt/diskb/penglong/zzff/Pretrained-IPT-main/IPT_sr4.pt --data_test Set5+Set14+Urban100+Manga109+B100 --scale 4 --test_only --save_results
python main.py --dir_data /mnt/diskb/penglong/zzff/Pretrained-IPT-main/data/dataset --pretrain /mnt/diskb/penglong/zzff/RCAN-master/RCAN_TestCode/model/RCAN_BIX4.pt --data_test Set5+Set14+Urban100+Manga109+B100 --scale 4 --test_only --save_results
  File "/mnt/diskb/penglong/zzff/RCAN-master/RCAN_TestCode/code/data/__init__.py", line 3, in <module>
    from dataloader import MSDataLoader
  File "/mnt/diskb/penglong/zzff/RCAN-master/RCAN_TestCode/code/dataloader.py", line 10, in <module>
    from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
ImportError: cannot import name '_update_worker_pids' from 'torch._C' (/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/_C.cpython-39-x86_64-linux-gnu.so)

python main.py --data_test Set5 --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5

    from torch.utils.data.dataloader import _worker_manager_loop
ImportError: cannot import name '_worker_manager_loop'

python main.py --data_test Manga109 --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --save 'RCAN' 
python main.py --model san --data_test Manga109 --save SAN --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath 'your path' --testset Set5 --pre_train ../model/SAN_BIX4.pt
