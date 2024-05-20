FROM tensorflow/tensorflow:2.16.1-gpu

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

USER root

RUN pip install pygame==2.5.2 mutagen==1.47.0 matplotlib==3.8.4 pandas==2.2.2 ipykernel==6.29.4
RUN pip install osrparse==7.0.0 scikit-learn==1.4.2

WORKDIR /home/root
