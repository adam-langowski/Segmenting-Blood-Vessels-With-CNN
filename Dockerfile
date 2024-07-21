FROM python:3.12.1

RUN apt-get update && apt-get install -y \
    python3-pil.imagetk \
    python3-tk \
    libgl1-mesa-glx

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "App_SegmentingVessels.py"]
