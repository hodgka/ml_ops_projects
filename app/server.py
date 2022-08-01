from datetime import datetime
from pickle import GLOBAL
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib
import os

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "./data/news_classifier.joblib"
        },
        "classes": [
                'Business',
                'Entertainment',
                'Health',
                'Music Feeds',
                'Sci/Tech',
                'Software and Developement',
                'Sports',
                'Toons'
            ]

    },
    "service": {
        "log_destination": "./data/logs.out"
    },
    
}

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.classes = self.config["classes"]
        """
        [TO BE IMPLEMENTED]
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer` (Hint: revisit Week 1 Step 4)
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`
        """
        try:
            featurizer_dim = config['featurizer']['sentence_transformer_embedding_dim']
            featurizer_model = config['featurizer']['sentence_transformer_model']
            sentence_transformer_model = SentenceTransformer(f'sentence-transformers/{featurizer_model}')
            featurizer = TransformerFeaturizer(featurizer_dim, sentence_transformer_model)
        except Exception as e:
            logger.error("Could not load valid featurizer config.")
        
        try:
            model = joblib.load(config['classifier']['serialized_model_path'])
        except Exception as e:
            logger.error("Could not load model.")

        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', model)
        ])
    
    def prep_input(self, input):
        return input['title'] + " " + input['description']

    def predict_proba(self, model_input: dict) -> dict:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization, 
        run model inference on a given model input, and return the 
        model prediction probability scores across all labels

        Output format: 
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2 
            ...
        }
        """
        input = self.prep_input(model_input)
        if not isinstance(input, list):
            input = [input]
        # print('[*]: ', input)
        results = self.pipeline.predict_proba(input)[0]
        # print("[!]: ", results)
        results = {cls: score for cls, score in zip(self.classes, results)}
        return results

    def predict_label(self, model_input: dict) -> str:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """
        input = model_input['description']
        if not isinstance(input, list):
            input = [input]
        preds = self.pipeline.predict(input)[0]
        return preds


app = FastAPI()
MODEL = None
log_file = None


@app.on_event("startup")
def startup_event():
    """
        [TO BE IMPLEMENTED]
        2. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass any relevant config parameters from `GLOBAL_CONFIG` that are needed by NewsCategoryClassifier 
        3. Open an output file to write logs, at the destimation specififed by GLOBAL_CONFIG['service']['log_destination']
        
        Access to the model instance and log file will be needed in /predict endpoint, make sure you
        store them as global variables
    """
    global MODEL
    global log_file 
    os.chdir('..')
    MODEL = NewsCategoryClassifier(GLOBAL_CONFIG["model"])

    log_file = logger.add(GLOBAL_CONFIG["service"]["log_destination"])    
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
        [TO BE IMPLEMENTED]
        1. Make sure to flush the log file and close any file pointers to avoid corruption
        2. Any other cleanups
    """
    global log_file
    logger.remove(log_file)
    # app.log.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
        [TO BE IMPLEMENTED]
        1. run model inference and get model predictions for model inputs specified in `request`
        2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
        {
            'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
            'request': dictionary representation of the input request,
            'prediction': dictionary representation of the response,
            'latency': time it took to serve the request, in millisec
        }
        3. Construct an instance of `PredictResponse` and return

        {
  "source": "BBC Technology",
  "url": "http://news.bbc.co.uk/go/click/rss/0.91/public/-/2/hi/business/4144939.stm",
  "title": "System gremlins resolved at HSBC",
  "description": "Computer glitches which led to chaos for HSBC customers on Monday are fixed, the High Street bank confirms."
}
    """
    global MODEL

    start = datetime.now()
    input_dict = request.dict()
    scores = MODEL.predict_proba(input_dict)
    label = MODEL.predict_label(input_dict)
    # logger.info(type(scores))
    # label = MODEL.get_label(scores)
    
    response = PredictResponse(scores=scores,label=label)
    # response = {
        # "scores": scores,
        # "label": label,
    # }
    end = datetime.now()
    request_data = {
        "timestamp":start,
        "request": input_dict,
        "prediction": response,
        "latency": end - start

    }
    logger.info(request_data)
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
