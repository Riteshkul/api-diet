from flask import Flask,request,jsonify
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route('/',methods=['GET'])
def index():
    target=int(request.args['target'])
    bmi=float(request.args['bmi'])
    bmr=float(request.args['bmr'])
    type_body=int(request.args['body_type'])
    bfp=float(request.args['bfp'])
    pred=model.predict(np.array([target,bmi,bmr,type_body,bfp]).reshape(1,-1))
    return jsonify(prediction=str(pred))
    



if __name__=="__main__":
    app.run(debug=True)
