import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import TFGPT2LMHeadModel, BertTokenizer
import tensorflow as tf
import re
from flask import Flask
from flask import render_template
from flask import request,  Response

app = Flask(__name__)
app.config['DEBUG'] = True
model = TFGPT2LMHeadModel.from_pretrained("gpt2-cn-50")

@app.route('/')
def index_main():
    return render_template('index.html')

@app.route('/random', methods=["GET", "POST"])
def get_text():
    if request.method == "GET":
        sentence = request.args.get("message")
        result = test_model(sentence)
        return Response(result)

def test_model(sentence):
    if " " not in sentence:
        sentence = re.sub("", " ", sentence)[1:]
    # print(sentence)
    tokenizer = BertTokenizer(vocab_file="vocab.txt")
    input_data = tokenizer([sentence], return_tensors="tf",add_special_tokens=False)

    input_ids = input_data["input_ids"][0].numpy()
    input_ids = list(input_ids)
    input_ids.insert(0, tokenizer.get_vocab()["[CLS]"])

    input_ids = tf.constant(input_ids)[None, :]
    # print(input_ids[0].numpy())
    # exit()
    for i in range(100):
        predictions = model(input_ids=input_ids, training=False)[0]

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id[0].numpy(), [3]):
            break

        input_ids = tf.concat([input_ids, predicted_id], axis=-1)
    # print(input_ids)
    result = "".join(tokenizer.batch_decode(tf.squeeze(input_ids, axis=0)))
    result = result.split("|")
    up_sentence = result[0]
    up_sentence = up_sentence.split("]")[1]
    un_sentence = result[1]
    # print("上联：", up_sentence)
    # print("下联：", un_sentence)
    result = "上联："+ up_sentence+"\n\n"+"下联："+ un_sentence
    return result

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
