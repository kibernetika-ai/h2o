#!/usr/bin/env bash

docker build -t kuberlab/h2o.ai:latest -f Dockerfile .
docker push kuberlab/h2o.ai:latest
