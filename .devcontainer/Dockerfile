FROM python:3.10.16-bullseye

COPY . /workspaces/Dev-Py
WORKDIR /workspaces/Dev-Py
# Install dependencies
RUN apt-get update && \
    apt-get install -y git wget


RUN apt-get update 
RUN ln -s /usr/local/bin/python /usr/local/bin/py

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir
