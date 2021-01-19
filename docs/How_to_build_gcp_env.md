# How To Build GCP Enviroment

- GCE 立ち上げ編
  - .env.example をコピーして環境変数を設定
  - make create-instance-preemtible or create-instance で instance を立ち上げる
  - mkae connect-instance で接続できることを確かめる
  - ssh 任意の接続名 で接続できることを確かめる
  - GPUを立ち上げている場合
    - `nvidia-smi` コマンドでGPUが読み込めていることを確かめる
- Github Private Repository への接続する設定
  - https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account
