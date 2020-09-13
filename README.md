## MedicalNER: Deploy PyTorch NER Model with Flask and Docker as Web App

A pretty and customizable web app to deploy your DL model with ease

## Usage Demo

1. Clone the repo

  ```git
  git clone https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp.git
  ```

2. Run the following instructions

  ```
  docker build -t ner-model .
  ```
  
  then
  
  ```
  docker run -d -p 8000:8000 ner-model
  ```

3. Open the following URL

  ```
  http://0.0.0.0:8000/apidocs
  ```
  
4. Input a medical sentence (in Chinese) and see the recognized entities!

<p align="center">
  <a href="https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp"> <img src="https://github.com/SuperBruceJia/pytorch-flask-deploy-webapp/raw/master/screenshot.png"></a> 
</p>

Have fun!

## Docker Image

The Docker Images have been uploaded to [Docker Hub](https://hub.docker.com/r/shuyuej/ner-pytorch-model).

## License

MIT License
