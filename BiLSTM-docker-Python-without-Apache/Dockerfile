FROM python:3.6-slim-buster

EXPOSE 8000

WORKDIR /var/www/NERmodel/

COPY ./NERmodel.py /var/www/NERmodel/NERmodel.py
COPY ./models /var/www/NERmodel/models/
COPY ./model.py /var/www/NERmodel/model.py
COPY ./utils.py /var/www/NERmodel/utils.py
COPY ./config.yml /var/www/NERmodel/config.yml

# Add Python Packages
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --ignore-installed --no-cache-dir PyYAML
RUN pip install --upgrade --no-cache-dir flask
RUN pip install --upgrade --no-cache-dir flasgger==0.8.1
RUN pip install --upgrade --no-cache-dir torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

CMD python /var/www/NERmodel/NERmodel.py
