FROM osgeo/gdal:ubuntu-full-3.6.3

WORKDIR /Hack_final

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y

RUN pip install -r requirements.txt

# RUN python app.py