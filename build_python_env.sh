# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
wait
source $HOME/.poetry/env

# Install pyenv
curl https://pyenv.run | bash
wait
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install python with pyenv.
pyenv install 3.9.0
wait
pyenv global 3.9.0
