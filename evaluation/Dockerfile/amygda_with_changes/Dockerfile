FROM ubuntu:18.04

# install basic dependencies
RUN apt update && apt install -y python3 python3-pip git libsm6 libxext6 libxrender-dev
RUN cp /usr/bin/python3 /usr/bin/python

# install cobs
ARG amygda_repo_URL
ARG commit_id_or_branch
RUN git clone --recursive $amygda_repo_URL && cd amygda && git checkout $commit_id_or_branch && pip3 install -r requirements.txt && chmod +x analyse-plate-with-amygda.py

# add amygda
ENV PATH="/amygda:${PATH}"

ENTRYPOINT ["/amygda/analyse-plate-with-amygda.py"]
CMD ["-h"]