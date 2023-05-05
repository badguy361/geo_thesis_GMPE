FROM python:3.8.16-buster
WORKDIR /TSMIP
COPY . .
ENV PATH="/TSMIP/machine_learning/design_pattern:${PATH}"
RUN pip3 install -r requirements.txt

CMD ["cd machine_learning/design_pattern", "python -m unittest discover tests"]