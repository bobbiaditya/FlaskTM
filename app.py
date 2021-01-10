from flask import Flask,render_template, request,jsonify
import pickle

import re
import numpy as np
import pandas as pd
import spacy
# import spacy_lookup
import json 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

import requests

app = Flask(__name__)
nlp = spacy.blank('id')
customize_stop_words=['terima','kasih','kepada','yth','terima kasih','dan','nya','mohon','tindak','lanjut','tindaklanjuti']
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        input_lang = request.form['input_lang']
        output_lang = request.form['output_lang']
        judul_raw = request.form['judul']
        isi_raw = request.form['isi']
        if(input_lang!='id'):
            data_input={
                "judul": '\"'+judul_raw+'\"',
                "isi": '\"'+isi_raw+'\"',
                "instansi" : '\"'+'blank'+'\"',
                "SourceLanguageCode": '\"'+input_lang+'\"',
                "TargetLanguageCode": '\"id\"'
            }
            data_translated=translation(data_input)
            judul = data_translated['judul_translated']
            isi = data_translated['isi_translated']
        else:
            judul = judul_raw
            isi = isi_raw
        nama_instansi = predict(judul,isi)
        print(judul,isi,nama_instansi)
        data={
                "judul": '\"'+judul+'\"',
                "isi": '\"'+isi+'\"',
                "instansi" : '\"'+nama_instansi+'\"',
                "SourceLanguageCode": '\"id\"',
                "TargetLanguageCode": '\"'+output_lang+'\"'
            }
        data_translated=translation(data)
        # data_tw={
        #         "judul": '\"'+judul+'\"',
        #         "isi": '\"'+isi+'\"',
        #         "instansi" : '\"'+nama_instansi+'\"',
        #         "SourceLanguageCode": '\"id\"',
        #         "TargetLanguageCode": '\"zh-tw\"'
        #     }   
        # data_translated_tw=translation(data_tw)                
        return render_template("hasil.html",judul=judul_raw,isi=isi_raw,data_translated=data_translated)

    else:
        return render_template("index.html")

def preproses(text):
    with open('indo_slang_word.txt') as f: 
        data = f.read() 
    indo_slang = json.loads(data)
    text = text.lower()
    text = ' '.join(list(map(indo_slang.get, text.split(), text.split())))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I|re.A) # if expression in the sentence is not a word then this code change them to space
    text = text.strip() # remove extra whitespace
    text = re.sub(' +', ' ', text)
    text = nlp(text)
    print('sebelum:',text)
    text = [token.lemma_ for token in text if not token.is_stop] 
    print('sesudah:',text)
    print('\n')
    return text

def tokenize_judul(judul_cleaned):
    with open('tokenizer_judul.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    judul_seq = tokenizer.texts_to_sequences([judul_cleaned])
    judul_seq = pad_sequences(judul_seq, padding='post',maxlen=10)
    return judul_seq

def tokenize_isi(isi_cleaned):
    with open('tokenizer_isi.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    isi_seq = tokenizer.texts_to_sequences([isi_cleaned])
    isi_seq = pad_sequences(isi_seq, padding='post',maxlen=20)
    return isi_seq

def predict(judul,isi):
    judul_cleaned = preproses(judul)
    judul_seq = tokenize_judul(judul_cleaned)

    isi_cleaned = preproses(isi)
    isi_seq = tokenize_isi(isi_cleaned)   

    model = keras.models.load_model('combined_model.h5')
    target_pred = model.predict([judul_seq,isi_seq])
    instance_id = np.argmax(target_pred,axis=1) + 1
    id_instansi = instance_id[0]
    nama_instansi = encode(id_instansi)
    return(nama_instansi)


def encode(id):
    encoding={
        1:"Badan Penyelenggara Jaminan Sosial Kesehatan",
        2:"Pemerintah Provinsi DKI Jakarta",
        3:"Kementerian Sosial",
        4:"Pemerintah Kota Bandung",
        5:"Pemerintah Kota Semarang",
        6:"Kementerian Pendidikan dan Kebudayaan",
        7:"Kementerian Badan Usaha Milik Negara",
        8:"Kementerian Hukum dan HAM",
        9:"Pemerintah Kabupaten Bojonegoro",
        10:"Kementerian Dalam Negeri",
        11:"Badan Kepegawaian Negara",
        12:"Kepolisian Republik Indonesia"
        } 
    return encoding[id]

def translation(data):
    judul="{\r\n"+"  \"judul\": "+data['judul']+",\r\n"
    isi="  \"isi\": "+data['isi']+",\r\n"
    instansi="  \"instansi\":"+data['instansi']+",\r\n"
    sourcelang=" \"SourceLanguageCode\":"+data['SourceLanguageCode']+",\r\n"
    targetlang=" \"TargetLanguageCode\":"+data['TargetLanguageCode']+"\r\n}"
    url = "https://8gmj10kr0a.execute-api.us-east-1.amazonaws.com/Dev1"
    headers = {
    'Content-Type': 'text/plain',
    'charset':'utf-8',
    }

    response = requests.request("GET", url, headers=headers, data=(judul+isi+instansi+sourcelang+targetlang).encode('utf-8'))

    return(response.json())

@app.route('/api/', methods=['GET'])
def api():
    query_parameters = request.args
    judul_raw=query_parameters['judul']
    isi_raw=query_parameters['isi']
    input_lang=query_parameters['input_lang']
    output_lang=query_parameters['output_lang']
    if(input_lang!='id'):
        data_input={
            "judul": '\"'+judul_raw+'\"',
            "isi": '\"'+isi_raw+'\"',
            "instansi" : '\"'+'blank'+'\"',
            "SourceLanguageCode": '\"'+input_lang+'\"',
            "TargetLanguageCode": '\"id\"'
        }
        data_translated=translation(data_input)
        judul = data_translated['judul_translated']
        isi = data_translated['isi_translated']
    else:
        judul = judul_raw
        isi = isi_raw
    nama_instansi = predict(judul,isi)
    data={
            "judul": '\"'+judul+'\"',
            "isi": '\"'+isi+'\"',
            "instansi" : '\"'+nama_instansi+'\"',
            "SourceLanguageCode": '\"id\"',
            "TargetLanguageCode": '\"'+output_lang+'\"'
        }
    data_translated=translation(data)
    print(data_translated)
    hasil_akhir={
        'data_input':{
            'Title':judul_raw,
            'Content':isi_raw,
            'Input Language':input_lang
        },
        'data_translated':{
            'Title':data_translated['judul_translated'],
            'Content':data_translated['isi_translated'],
            'Government Agency':data_translated['instansi_translated'],
            'Output Language':data_translated['TargetLanguageCode']
        }
    }
    return jsonify(hasil_akhir)

if __name__ == "__main__":
    app.run(debug=True)