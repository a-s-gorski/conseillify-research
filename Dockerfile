FROM pytorch/pytorch:latest
RUN apt-get update -y
RUN apt-get install -y gcc
COPY . .
ENV ARTIFACTS_PATH='artifacts/'
RUN pip3 install -r requirements.txt
RUN python3 serving/setup_nltk.py
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8081"]