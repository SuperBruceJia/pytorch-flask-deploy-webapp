## MedicalNER: Deploy PyTorch NER Model with Flask and Docker as Web App

A pretty and customizable web app to deploy your Deep Learning (DL) model with ease

## Notice

***This repo was to deploy any kind of ML or DL model(s) rather than just NER model.***

## Usage Demo

1. Clone the repo

    ```
    $ git clone https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp.git
    ```

2. Run the following instructions
  
    1). Build Docker Image

      ```
      $ docker build -t ner-model .
      ```
  
    2). Make and Run a container for the above Image
  
    ```
    $ docker run -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -it --rm -d -p 8000:8000 ner-model
    
    or 
    
    $ docker run -d -p 8000:8000 ner-model
    ```
  
3. Open the following URL (Google Chrome is recommended)  
  
    ```html
    http://0.0.0.0:8000/apidocs/#!/default/get_predict
    
    or 
    
    http://0.0.0.0:8000/apidocs
    ```
  
4. Input a medical sentence (in Chinese) and see the recognized entities!

    <p align="center">
      <a href="https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp"> <img src="https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/raw/master/screenshot.png"></a> 
    </p>

    🏆 Enjoy your journey!

## Docker Image

The Docker Images have been uploaded to [Docker Hub](https://hub.docker.com/r/shuyuej/ner-pytorch-model/tags).

## Size of the Docker Image

1. [Anaconda Python Environment](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/tree/master/BiLSTM-docker-Anaconda) --> 2.22 GB (V1.0)

    *Used Image*: continuumio/anaconda3:4.4.0

2. [Standard Python Environment](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/tree/master/BiLSTM-docker-Python) --> 617.96 MB (V2.0)

    *Used Image*: Python:3
    
3. [Smaller Python Environment](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/tree/master/BiLSTM-docker-Python-Smaller) --> 447.05 MB (V4.0)

    *Used Image*: python:3.8-slim-buster

## Reference Image and Readings

1. [Python Image](https://hub.docker.com/_/python?tab=description)

2. [Anaconda Image](https://hub.docker.com/r/continuumio/anaconda3)

3. [My uploaded Docker Images](https://hub.docker.com/r/shuyuej/ner-pytorch-model/tags)

4. [The best Docker base image for your Python application](https://pythonspeed.com/articles/base-image-python-docker-images/)

5. [Docker Container UTF-8 Encoding](https://developer.aliyun.com/article/175738)

## License

MIT License
