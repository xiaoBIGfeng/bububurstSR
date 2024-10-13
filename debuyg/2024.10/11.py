>conda config --set proxy_servers.http http://proxy.huawei.com:8080
>conda config --set proxy_servers.https http://proxy.huawei.com:8080
>conda config --set proxy_servers.ssl_verify fals

Collecting package metadata (current_repodata.json): failed

CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/current_repodata.json>
Elapsed: -

An HTTP error occurred when trying to retrieve this URL.
HTTP errors are often intermittent, and a simple retry will get you on your way.
'https//mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64'
Retrieving notices: ...working... ERROR conda.notices.fetch:get_channel_notice_response(68): Request error <HTTPSConnectionPool(host='mirrors.tuna.tsinghua.edu.cn', port=443): Max retries exceeded with url: /anaconda/pkgs/free/notices.json (Caused by ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 502 Parent proxy unreacheable')))> for channel: anaconda/pkgs/free url: https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/notices.json
ERROR conda.notices.fetch:get_channel_notice_response(68): Request error <HTTPSConnectionPool(host='repo.anaconda.com', port=443): Max retries exceeded with url: /pkgs/main/notices.json (Caused by ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 502 Parent proxy unreacheable')))> for channel: defaults url: https://repo.anaconda.com/pkgs/main/notices.json
ERROR conda.notices.fetch:get_channel_notice_response(68): Request error <HTTPSConnectionPool(host='repo.anaconda.com', port=443): Max retries exceeded with url: /pkgs/r/notices.json (Caused by ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 502 Parent proxy unreacheable')))> for channel: defaults url: https://repo.anaconda.com/pkgs/r/notices.json
done
Collecting package metadata (current_repodata.json): failed

ProxyError: Conda cannot proceed due to an error in your proxy configuration.
Check for typos and other configuration errors in any '.netrc' file in your home directory,
any environment variables ending in '_PROXY', and any other system-wide proxy
configuration settings.




['001_0291']
base: 29.201473
0代1: 29.301529 True
0代2: 29.255836 True
0代3: 29.303669 True
0代4: 29.262743 True
0代5: 29.292763 True
0代6: 29.251917 True
0代7: 29.294922 True
0代8: 29.252796 True
0代9: 29.284613 True
0代10: 29.256308 True
0代11: 29.325724 True
0代12: 29.285898 True
0代13: 29.390007 True

0代2: 29.255836 True
1代2: 29.215881 True
3代2: 29.217005 True
4代2: 29.221323 True
5代2: 29.208765 True
6代2: 29.21849 True
7代2: 29.215637 True
8代2: 29.21696 True
9代2: 29.220533 True
10代2: 29.215752 True
11代2: 29.19955 False
12代2: 29.20993 True
13代2: 29.205656 True

step 79 ：['001_0292']
base: 35.72798
0代1: 35.57646 False
0代2: 35.88121 True
0代3: 35.88059 True
0代4: 35.928024 True
0代5: 35.96012 True
0代6: 35.936443 True
0代7: 35.803867 True
0代8: 35.91766 True
0代9: 35.719765 False
0代10: 35.843895 True
0代11: 35.878582 True
0代12: 35.93841 True
0代13: 35.781376 True
数据集dataset_manual 添加LR路径我改成了添加未对齐图片的路径
OpenBLAS blas_thread_init: pthread_create failed for thread 3 of 64: Resource temporarily unavailable
OpenBLAS blas_thread_init: RLIMIT_NPROC 3088023 current, 3088023 max
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
libgomp: Thread creation failed: Resource temporarily unavailable
['029_0370'] 1535
base: 27.589643
2代1: 27.11208 False
2代3: 26.837772 False  
2代4: 26.774208 False
2代5: 26.612476 False
2代6: 26.148024 False
2代7: 25.839651 False
2代8: 25.741543 False
2代9: 25.586098 False
2代10: 24.743288 False
2代11: 24.706772 False
2代12: 24.679462 False
2代13: 24.594707 False
2代其他帧都会有下降

['029_0370']
base: 27.589643
1代2: 28.211937 True
1代3: 28.297562 True
1代4: 28.328781 True
1代5: 28.317875 True
1代6: 28.285545 True
1代7: 28.059414 True
1代8: 27.964432 True
1代9: 27.85055 True
1代10: 27.09 False
1代11: 26.9504 False
1代12: 26.763954 False
1代13: 26.735928 False


['000_0436']
base: 27.627237
2代1: 27.622526 False
2代3: 27.621181 False
2代4: 27.61655 False
2代5: 27.614372 False
2代6: 27.607628 False
2代7: 27.606113 False
2代8: 27.605736 False
2代9: 27.599152 False
2代10: 27.60067 False
2代11: 27.590904 False
2代12: 27.585487 False
2代13: 27.379314 False
['000_0436']
base: 27.627237
3代1: 27.622238 False
3代2: 27.617855 False
3代4: 27.615189 False
3代5: 27.611835 False
3代6: 27.60153 False
3代7: 27.597443 False
3代8: 27.593807 False
3代9: 27.586685 False
3代10: 27.57907 False
3代11: 27.563122 False
3代12: 27.561558 False
3代13: 27.349312 False
['000_0436']
base: 27.627237
4代1: 27.62283 False
4代2: 27.618298 False
4代3: 27.619942 False
4代5: 27.617668 False
4代6: 27.607014 False
4代7: 27.602903 False
4代8: 27.597855 False
4代9: 27.592508 False
4代10: 27.581696 False
4代11: 27.565895 False
4代12: 27.567062 False
4代13: 27.347343 False
['000_0436']
base: 27.627237
5代1: 27.623827 False
5代2: 27.623684 False
5代3: 27.6223 False
5代4: 27.618628 False
5代6: 27.612383 False
5代7: 27.61082 False
5代8: 27.611755 False
5代9: 27.60685 False
5代10: 27.608717 False
5代11: 27.599077 False
5代12: 27.594995 False
5代13: 27.393204 False

['000_0436']
base: 27.627237 最好的帧
6代1: 27.633244 True
6代2: 27.633198 True
6代3: 27.636608 True
6代4: 27.637434 True
6代5: 27.645096 True
6代7: 27.64235 True
6代8: 27.640247 True
6代9: 27.644537 True
6代10: 27.637003 True
6代11: 27.637632 True
6代12: 27.63029 True
6代13: 27.423233 False
