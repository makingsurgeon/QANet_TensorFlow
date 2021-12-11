import QANet_keras as QANet
import os
from keras.optimizers import Adam
import numpy as np
from keras.models import load_model
import pandas as pd
import pickle
from keras.callbacks import Callback
import argparse
from layers.ExponentialMovingAverage import ExponentialMovingAverage
import keras.backend as K
import collections
from utils.output import write_predictions
from utils.evaluation import evaluate
from preprocess import SquadExample, InputFeatures

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default='./dataset_wordpiece/trainset_wordpiece.pkl', type=str, help="train path")
parser.add_argument("--dev_path", default='./dataset_wordpiece/devset_wordpiece.pkl', type=str, help="dev path")
parser.add_argument("--word_embedding", default='./dataset_wordpiece/word_emb_mat.npy', type=str, help="word embedding path")
parser.add_argument("--char_embedding", default='./dataset_wordpiece/char_emb_mat.npy', type=str, help="char embedding path")
parser.add_argument("--word_dim", default=300, type=int, help="dim of glove word vector")
parser.add_argument("--char_dim", default=64, type=int, help="dim of character")
parser.add_argument("--cont_limit", default=384, type=int, help="context word limit")
parser.add_argument("--ques_limit", default=64, type=int, help="question word limit")
parser.add_argument("--char_limit", default=16, type=int, help="char limit in one word")
parser.add_argument("--ans_limit", default=30, type=int, help="answer word limit")
parser.add_argument("--filters", default=128, type=int, help="filters")
parser.add_argument("--num_head", default=8, type=int, help="head num for attentionn")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout rate")
parser.add_argument("--batch_size", default=24, type=int, help="batch size")
parser.add_argument("--epoch", default=25, type=int, help="epochs")
parser.add_argument("--ema_decay", default=0.9999, type=float, help="ema decay")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument("--warm_up_steps", default=1000, type=int, help="warm up steps")
parser.add_argument("--name", default='QANet_wordpiece', type=str, help="saving name of the model")
parser.add_argument("--use_cove", default=False, type=bool, help="whether to use cove")
parser.add_argument("--cove_path", default='model/Keras_CoVe_nomask.h5', type=str, help="Cove path")

config = parser.parse_args()
config.path = 'model/' + config.name

os.makedirs(config.path, exist_ok=True)

# load trainset
with open(config.train_path, 'rb') as f:
    train_data = pickle.load(f)
train_data['start_label_fin'] = np.argmax(train_data['y_start'], axis=-1)
train_data['end_label_fin'] = np.argmax(train_data['y_end'], axis=-1)

# load valset
with open(config.dev_path, 'rb') as f:
    dev_data = pickle.load(f)

with open('./dataset_wordpiece/dev_examples.pkl', 'rb') as f:
    eval_examples = pickle.load(f)
with open('./dataset_wordpiece/dev_features.pkl', 'rb') as f:
    eval_features = pickle.load(f)

# load embedding matrix
word_mat = np.load(config.word_embedding)
char_mat = np.load(config.char_embedding)

ems = []
f1s = []
cove_model = None
if config.use_cove:
    cove_model = load_model(config.cove_path)
    for layer in cove_model.layers:
        layer.trainable = False

model = QANet.QANet(config, word_mat=word_mat, char_mat=char_mat, cove_model=cove_model)
model.summary()

optimizer = Adam(lr=config.learning_rate, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
              loss_weights=[0.5, 0.5, 0, 0])

RawResult = collections.namedtuple("RawResult",
                                   ["qid", "start_logits", "end_logits"])


class QANet_callback(Callback):
    def __init__(self):
        self.global_step = 1
        self.max_f1 = 0
        self.keras_ema = ExponentialMovingAverage(model, decay=config.ema_decay,
                                                  temp_model=os.path.join(config.path, 'temp_model.h5'), type='cpu')
        super(Callback, self).__init__()

    def on_train_begin(self, logs=None):
        lr = min(config.learning_rate,
                 config.learning_rate / np.log(config.warm_up_steps) * np.log(self.global_step))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = min(config.learning_rate,
                 config.learning_rate / np.log(config.warm_up_steps) * np.log(self.global_step))
        K.set_value(self.model.optimizer.lr, lr)
        self.keras_ema.average_update()

    def on_epoch_end(self, epoch, logs=None):
        self.keras_ema.assign_shadow_weights()
        logits1, logits2, _, _ = self.model.predict(x=[dev_data['context_id'], dev_data['question_id'],
                                                       dev_data['context_char_id'], dev_data['question_char_id']],
                                                    batch_size=config.batch_size,
                                                    verbose=1)
        all_results = []
        for i, qid in enumerate(dev_data['qid']):
            start_logits = logits1[i, :]
            end_logits = logits2[i, :]
            all_results.append(RawResult(qid=qid,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
        output_prediction_file = os.path.join(config.path, 'output_prediction.json')
        output_nbest_file = os.path.join(config.path, 'output_nbest.json')
        write_predictions(eval_examples, eval_features, all_results,
                          n_best_size=20, max_answer_length=config.ans_limit,
                          do_lower_case=False, output_prediction_file=output_prediction_file,
                          output_nbest_file=output_nbest_file)
        metrics = evaluate('original_data/dev-v1.1.json', output_prediction_file, None)
        ems.append(metrics['exact'])
        f1s.append(metrics['f1'])
        result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
        result.to_csv('logs/result_' + config.name + '.csv', index=None)
        if f1s[-1] > self.max_f1:
            self.max_f1 = f1s[-1]
            model.save_weights(os.path.join(config.path, 'QANet_model_' + config.name + '.h5'))
        model.load_weights(self.keras_ema.temp_model)


qanet_callback = QANet_callback()
qanet_callback.set_model(model)



model.fit(x=[train_data['context_id'], train_data['question_id'],
             train_data['context_char_id'], train_data['question_char_id']],
          y=[train_data['y_start'], train_data['y_end'], train_data['start_label_fin'],
             train_data['end_label_fin']],
          batch_size=config.batch_size,
          epochs=config.epoch,
          callbacks=[qanet_callback])



