FROM manifoldai/orbyter-ml-dev:1.1
RUN pip install gym
RUN pip install fire
RUN pip install tensorflow
RUN pip install tensorboard
RUN pip install keras
RUN pip install torch
RUN pip install torchvision
RUN pip install gym
RUN pip install cufflinks
RUN pip install tensorboardX
RUN pip install gym[atari]
RUN pip install git+https://github.com/JKCooper2/gym-bandits#egg=gym-bandits
