FROM python:3.8.16-buster
WORKDIR /app
COPY . .

RUN pip3 install -r requirements.txt

CMD ["/bin/sh"]