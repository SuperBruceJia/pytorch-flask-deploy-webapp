#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary Python Packages
import re
import pickle
import torch
import yaml
from model import BiLSTMCRF
from utils import *
import warnings
import numpy as np
from flask import Flask, request
from flasgger import Swagger


warnings.filterwarnings("ignore")
device = torch.device("cpu")


app = Flask(__name__)
swagger = Swagger(app)


def load_params(path: str):
    """
    Load the parameters (data)
    """
    with open(path + "data.pkl", "rb") as fopen:
        data_map = pickle.load(fopen)
    return data_map


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def cut_text(text, length):
    textArr = re.findall('.{' + str(length) + '}', text)
    textArr.append(text[(len(textArr) * length):])
    return textArr


def load_config():
    """
    Load hyper-parameters from the YAML file
    """
    fopen = open("config.yml")
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()
    return config


class ChineseNER:
    def __init__(self, entry="train"):
        # Load some Hyper-parameters
        config = load_config()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.dropout = config.get("dropout")
        self.tags = config.get("tags")
        self.learning_rate = config.get("learning_rate")
        self.epochs = config.get("epochs")
        self.weight_decay = config.get("weight_decay")
        self.transfer_learning = config.get("transfer_learning")
        self.lr_decay_step = config.get("lr_decay_step")
        self.lr_decay_rate = config.get("lr_decay_rate")
        self.max_length = config.get("max_length")

        # Model Initialization
        self.main_model(entry)

    def main_model(self, entry):
        """
        Model Initialization
        """
        # The Testing & Inference Process
        if entry == "predict":
            data_map = load_params(path=self.model_path)
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")
            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                dropout=0.0,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
                max_length=self.max_length
            )
            self.restore_model()

    def restore_model(self):
        """
        Restore the model if there is one
        """
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("Model Successfully Restored!")
        except Exception as error:
            print("Model Failed to restore! {}".format(error))

    def predict(self, input_str):
        """
        Prediction & Inference Stage
        :param input_str: Input Chinese sentence
        :return entities: Predicted entities
        """
        if len(input_str) != 0:
            # Full-width to half-width
            input_str = strQ2B(input_str)
            input_str = re.sub(pattern='。', repl='.', string=input_str)
            text = cut_text(text=input_str, length=self.max_length)

            cut_out = []
            for cuttext in text:
                # Get the embedding vector (Input Vector) from vocab
                input_vec = [self.vocab.get(i, 0) for i in cuttext]

                # convert it to tensor and run the model
                sentences = torch.tensor(input_vec).view(1, -1)

                length = np.expand_dims(np.shape(sentences)[1], axis=0)
                length = torch.tensor(length, dtype=torch.int64, device=device)

                _, paths = self.model(sentences=sentences, real_length=length, lengths=None)

                # Get the entities from the model
                entities = []
                for tag in self.tags:
                    tags = get_tags(paths[0], tag, self.tag_map)
                    entities += format_result(tags, cuttext, tag)

                # Get all the entities
                all_start = []
                for entity in entities:
                    start = entity.get('start')
                    all_start.append([start, entity])

                # Sort the results by the "start" index
                sort_d = [value for index, value in sorted(enumerate(all_start), key=lambda all_start: all_start[1])]

                if len(sort_d) == 0:
                    return print("There was no entity in this sentence!!")
                else:
                    sort_d = np.reshape(np.array(sort_d)[:, 1], [np.shape(sort_d)[0], 1])
                    cut_out.append(sort_d)
            return cut_out
        else:
            return print('Invalid input! Please re-input!!\n')


@app.route('/predict', methods=["GET"])
def predict_iris_file():
    """Named Entity Recognition (NER) Prediction for Medical Services
    ---
    parameters:
      - name: input_str
        in: query
        type: string
        required: true
    """
    input_str = request.args.get("input_str")
    cn = ChineseNER("predict")
    prediction = cn.predict(input_str)
    return str(prediction)


# main function
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
