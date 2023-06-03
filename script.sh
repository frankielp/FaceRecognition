
#!/bin/bash

# Create a new Conda environment
conda create -y -n cinnamon

# Activate the Conda environment
source activate cinnamon

# Set an alias for Python 3
alias python=python3

# Download pretrained facenet
cd model
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-' -O- | sed -rn 's/.confirm=([0-9A-Za-z]+)._/\1\n/p')&id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-" -O pretrained.zip && rm -rf /tmp/cookies.txt
unzip pretrained.zip
mv 20180402-114759 pretrained
cd ..

# Install the requirements from the 'model/requirements.txt' file
pip3 install -r model/requirements.txt

# Run the 'app.py' script
python3 app.py

# Deactivate the Conda environment
conda deactivate

