from flask import Flask, render_template, request
from werkzeug import secure_filename
from PIL import Image
from numpy import array
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder 

#path='D:/classifier/myVariables.pickle'
path='myVariables.pickle'
#name='__main__'
name='__home__'
pickle_off = open(path,"rb")
myVariables=pickle.load(pickle_off)

W_1=myVariables[0]
W_2=myVariables[1] 
B_1=myVariables[2]
width=myVariables[3]
height=myVariables[4]
total=myVariables[5]

le=LabelEncoder()   
le.fit_transform(['calavera','mujer','monstruo']) 
 

app = Flask(__name__)  
app.config['MAX_CONTENT_PATH']= 20  
 
@app.route('/')
def upload_file():  
    return render_template('upload.html') 

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST': 
        f = Image.open(request.files['file'].stream)
        f=f.resize((width,height))
        f=array(f)		
        #f.save(secure_filename(f.filename))
        prueba_vacio1=[] 
#convierte un arreglo de listas en una sola lista de la imagen de prueba	
        for sublist1 in array(f):
            for sublist2 in sublist1:
                for sublist3 in sublist2:
                    prueba_vacio1.append(sublist3)
        prueba_vacio1=np.float32(prueba_vacio1)            
        test_X=np.array([prueba_vacio1]).reshape(1,total) 
        first_layer = tf.nn.sigmoid(tf.matmul( test_X,W_1)+B_1)
        YHAT = tf.argmax(tf.sigmoid(tf.matmul(first_layer,W_2)),axis=1)
        sess=tf.Session()
        prediccion=sess.run(YHAT)
        clasificacion=le.inverse_transform(prediccion[0])
        return render_template('output.html',prediccion=str(clasificacion))
    else: 
        return "IMAGE COULDN'T BE PREDICTED :("	
            
if __name__ == name:
    app.run(debug = True)