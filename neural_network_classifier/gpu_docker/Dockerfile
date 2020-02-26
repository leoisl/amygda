ARG TF_VERSION=2.1.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3-jupyter
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
COPY ./requirements.txt ./
RUN pip install -r requirements.txt