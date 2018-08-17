FROM alpine:latest
LABEL maintainer="Leonardo Alfaya Fonseca"

COPY ./machine_learning /machine_learning
COPY ./requirements.txt /machine_learning

RUN apk --no-cache add g++ \
        python3 \
        python3-dev &&\
        pip3 install --upgrade pip && \
        pip3 install jupyter && \
        pip3 install -r /machine_learning/requirements.txt

CMD jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root
