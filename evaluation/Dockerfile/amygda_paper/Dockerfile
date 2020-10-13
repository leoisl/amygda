FROM ubuntu:18.04

# install basic dependencies
RUN apt update && apt install -y python3 python3-pip git libsm6 libxext6 libxrender-dev
RUN cp /usr/bin/python3 /usr/bin/python
RUN cp /usr/bin/pip3 /usr/bin/pip

# install cobs
ARG amygda_repo_URL
ARG commit_id_or_branch
COPY ./requirements.txt ./
RUN git clone --recursive $amygda_repo_URL && cd amygda && git checkout $commit_id_or_branch && pip install -r /requirements.txt && python setup.py install

# add amygda
ENTRYPOINT ["/amygda/bin/analyse-plate-with-amygda.py"]
CMD ["-h"]