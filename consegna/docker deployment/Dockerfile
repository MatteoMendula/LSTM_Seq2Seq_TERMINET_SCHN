# syntax=docker/dockerfile:1

FROM python:3.7.15-bullseye

WORKDIR /python-docker

COPY . .
RUN pip3 install -r requirements.txt


CMD [ "python3", "serve_model.py " ]