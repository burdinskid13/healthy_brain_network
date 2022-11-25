
import os

os.system("rsync -avrz --include='*/' --include='classifier*.json'--include='classifier*importance*.csv' --include=*'classifier*performance*.csv' --exclude='*' maedbh@openmind-dtn.mit.edu:/om2/user/maedbh/hbn_data/interim/models/ /Users/maedbhking/Documents/healthy_brain_network/data/interim/models/")