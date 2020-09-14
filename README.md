## MedicalNER: Deploy PyTorch NER Model with Flask and Docker as Web App

A pretty and customizable web app to deploy your Deep Learning (DL) model with ease

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
    ```

    or 

    ```
    $ docker run -d -p 8000:8000 ner-model
    ```
  
3. Open the following URL (Google Chrome is recommended)  
  
    ```html
    http://0.0.0.0:8000/apidocs/#!/default/get_predict
    ```

    or 

    ```html
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

1. [Anaconda Python Environment](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/tree/master/BiLSTM-docker-Anaconda) -> 2.22 GB

    continuumio/anaconda3:4.4.0

2. [Standard Python Environment](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/tree/master/BiLSTM-docker-Python) -> 617.96 MB

    Python:3
    
3. [Smaller Python Environment](https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/tree/master/BiLSTM-docker-Python) -> 617.96 MB

    python:3.8-slim-buster
    
## License

MIT License
