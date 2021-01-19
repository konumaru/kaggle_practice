# How To Build GCP Enviroment

- GCE 立ち上げ編
  - .env.example をコピーして環境変数を設定
  - make create-instance-preemtible or create-instance で instance を立ち上げる
  - mkae connect-instance で接続できることを確かめる
  - ssh 任意の接続名 で接続できることを確かめる
  - GPUを立ち上げている場合
    - `nvidia-smi` コマンドでGPUが読み込めていることを確かめる
- Github Private Repository への接続する設定
  - https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
  - https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-g\ithub-account
- Dataのダウンロード
  - GCSから
  - scpコマンドで

## Setup The GCE Environment

```
# Make Copy and set .env
$ cp .env.example .env

# Setup .env
$ vim .env

# Build instance on GCP
$ make create-instance-preemptible

# If you don't want to use a Preemtible instance, you can do the following.
$ make create-instance

# Connect to the instance and verify that the GPU Driver is installed.
$ make connect-instance
(gce-instance) $ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.27.04    Driver Version: 460.27.04    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:04.0 Off |                    0 |
| N/A   50C    P8    10W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```


## Setup Github Private Repository
https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent



## Downlaod Data
### From Local to Cloud Instance
```
# File copy.
$ scp local_file_path host_name:remote_file_path

# Directory copy
$ scp -r local_file_path host_name:remote_file_path
```


### From Google Cloud Bucket to Cloud Instance

```
$ gsutil -m cp -r gs://{my-bucket} ./data/
```
