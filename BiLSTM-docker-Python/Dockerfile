FROM python:3
EXPOSE 8000

RUN apt-get update && apt-get install -y apache2 \
    apache2-dev \
    vim \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/www/NERmodel/

COPY ./NERmodel.py /var/www/NERmodel/NERmodel.py
COPY ./NERmodel.wsgi /var/www/NERmodel/NERmodel.wsgi
COPY ./models /var/www/NERmodel/models/
COPY ./model.py /var/www/NERmodel/model.py
COPY ./utils.py /var/www/NERmodel/utils.py
COPY ./requirements.txt /var/www/NERmodel/requirements.txt
COPY ./config.yml /var/www/NERmodel/config.yml

RUN pip install --upgrade pip
RUN pip install --upgrade --ignore-installed PyYAML
RUN pip install torch==1.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN mod_wsgi-express install-module
RUN mod_wsgi-express setup-server NERmodel.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80

CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND