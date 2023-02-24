FROM python:3.8.16-buster
WORKDIR /TSMIP
COPY . .

RUN pip3 install -r requirements.txt

CMD ["/bin/sh"]