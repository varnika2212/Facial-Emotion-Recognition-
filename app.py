from flask import Flask,render_template,request
import cv2
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
app=Flask(__name__,static_url_path='/static/')
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1
@app.route('/')
def index():
    print("success")
    return render_template('index.html')

@app.route('/after',methods=['GET','POST'])
def after():
    img=request.files['file1']
    img.save('static/file.jpg')
######################################################################
   # start the webcam feed
    img1 = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', img1)

    try:
        cv2.imwrite('static/cropped.jpg', cropped)

    except:
        pass
######################################################################
    try:
        image = cv2.imread('static/cropped.jpg', 0)
    except:
        image = cv2.imread('static/file.jpg', 0)
    image=cv2.imread('static/file.jpg',0)
    image=cv2.resize(image,(48,48))
    image=image/255.0
    image=np.reshape(image,(1,48,48,1))
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    prediction=loaded_model.predict(image)
    label_dict = {0:'Angry',1:'Disgusted',2:'Afraid',3:'Happy',4:'Neutral',5:'Sad',6:'Surprised'}
    prediction=np.argmax(prediction)
    final_prediction=label_dict[prediction]
    return render_template('after.html',data=final_prediction)


if __name__=="__main__":
    app.run(debug=True,port=8080)
