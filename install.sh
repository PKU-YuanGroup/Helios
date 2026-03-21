pip install -r requirements.txt

rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*

pip uninstall triton torchao xformers wandb tensorflow tensorflow-cpu -y
pip install wandb==0.23.0 triton==3.6.0

rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*
