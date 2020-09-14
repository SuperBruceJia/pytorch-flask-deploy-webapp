FROM python:3.8-slim-buster

EXPOSE 8000

RUN apt-get update && apt-get install -y apache2 apache2-dev

# RUN apt-get update && apt-get install -y apache2 \
#     apache2-dev \
#     locales \ 
#     locales-all \ 
#     && rm -rf /var/lib/apt/lists/* 

# RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
#     && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
# ENV LANG en_US.utf8

WORKDIR /var/www/NERmodel/

COPY ./NERmodel.py /var/www/NERmodel/NERmodel.py
COPY ./NERmodel.wsgi /var/www/NERmodel/NERmodel.wsgi
COPY ./models /var/www/NERmodel/models/
COPY ./model.py /var/www/NERmodel/model.py
COPY ./utils.py /var/www/NERmodel/utils.py
COPY ./config.yml /var/www/NERmodel/config.yml

# Add Python Packages
RUN pip install --upgrade pip
RUN pip install --upgrade --ignore-installed PyYAML
RUN pip install --upgrade flask
RUN pip install --upgrade flasgger==0.8.1
RUN pip install --upgrade mod_wsgi
RUN pip install --upgrade torch==1.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# RUN mod wsgi
RUN mod_wsgi-express install-module
RUN mod_wsgi-express setup-server NERmodel.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80

CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND