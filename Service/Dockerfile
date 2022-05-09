FROM python:3.8.0-slim-buster
ENV DEBIAN_FRONTEND noninteractive    # export DEBIAN_FRONTEND="noninteractive"
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8989
CMD [ "python3", "app.py" ]