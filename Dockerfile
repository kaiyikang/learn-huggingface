FROM python:3.9

RUN pip install --upgrade pip

COPY requirements.txt /

RUN pip install -r requirements.txt

WORKDIR /app

CMD ["/bin/bash"]


