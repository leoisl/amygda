ARG TF_VERSION=2.1.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3-jupyter
COPY ./requirements.txt ./
RUN pip install -r requirements.txt