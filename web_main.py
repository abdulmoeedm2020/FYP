from   yolov5                     import Darknet
from   camera                     import LoadStreams
from   flask                      import Flask,render_template,flash, redirect,url_for,session,logging,request,Response
from   extraFunction              import ipCaoncatenate
from   utils.general              import non_max_suppression, scale_coords, check_imshow
from   functools                  import wraps
from   datetime                   import datetime
from   social_Distecnce_detection import gen
import torch
import time
import cv2
import pyrebase
import os
import random
import json

config = {
  "apiKey": "AIzaSyBGG7G5f22LFGxrF8TmySLlHxvvcAaDvm0",
  "authDomain": "flask-7d1de.firebaseapp.com",
  "projectId": "flask-7d1de",
  "storageBucket": "flask-7d1de.appspot.com",
  "messagingSenderId": "472033700104",
  "appId": "1:472033700104:web:3e8860a19aecfe689b64bd",
  "measurementId": "G-GMN0QLLE0K",
  "databaseURL": "https://flask-7d1de-default-rtdb.firebaseio.com"

}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database();


app = Flask(__name__)
#secret key for the session
app.secret_key = os.urandom(24)

def isAuthenticated(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        #check for the variable that pyrebase creates
        if not auth.current_user != None:
            return redirect(url_for('signup'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    totalAccidentCount = 0
    count = db.child("count").get().val()
    for i in range(count):
        result = db.child("DetectionResult").child(i).get()
        rows  = result.val()
        Detection = rows["Detection"]
        # print(rows["Detection"])
        if Detection == 'Accidents' or  Detection == 'Accident':
            totalAccidentCount  =  totalAccidentCount + int(rows["Incident"])

        if Detection == 'person' or  Detection == 'persons':
            totalPersonCount  =  totalAccidentCount + int(rows["Incident"])
    return render_template("index.html", totalAccidentCount = totalAccidentCount, totalPersonCount = totalPersonCount)


@app.route("/logout")
def logout():
    auth.current_user = None
    session.clear()
    return redirect("/")



random.seed() 
@app.route("/graph")
def graph():
    return render_template("graph.html")

random.seed() 
@app.route("/accidentGraph")
def accidentGraph():
    return render_template("accidentGraph.html")

random.seed() 
@app.route("/graphPerson")
def graphPerson():
    return render_template("graphPerson.html")


def generate_random_data():
    incident = []
    time = []
    i=0    
    count = db.child("count").get().val()
    gCount = count - 2
    print(gCount)
    for gCount in range(count):

        result = db.child("DetectionResult").child(gCount).get()
        count = db.child("count").get().val()
        gCount = count + 1
        
        rows  = result.val()
        Detection = rows["Detection"]
        # print(rows["Detection"])
        if Detection == 'person' or Detection == 'persons' or Detection =='Injured Person' or Detection ==' ':
            incident  = rows["Incident"]
            time = rows["Time"]
            json_data = json.dumps(
                {
                "time": time,
                "value": incident,
                }
            )
            yield f"data:{json_data}\n\n"
    time.sleep(1)

def people():
    incident = []
    time = []
    i=0
    count = db.child("count").get().val()
    for i in range(count):
        result = db.child("DetectionResult").child(i).get()
        rows  = result.val()
        Detection = rows["Detection"]
        # print(rows["Detection"])
        if Detection == 'Accident' or  Detection == '' or Detection == 'Accidents':
            incident  = rows["Incident"]
            time = rows["Time"]
            json_data = json.dumps(
                {
                "time": time,
                "value": incident,
                }
            )
            yield f"data:{json_data}\n\n"

@app.route("/chart-data")
def chart_data():
    return Response(generate_random_data(),mimetype="text/event-stream")

@app.route("/people-data")
def people_data():
    return Response(people(), mimetype="text/event-stream")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
      email = request.form["mail"]
      password = request.form["passw"]
      try:
        user = auth.sign_in_with_email_and_password(email, password)
        user_id = user['idToken']
        user_email = email
        session['usr'] = user_id
        session["email"] = user_email
        return redirect("ipAddress")  
      except:
        return render_template("login.html", message="Wrong Credentials" )       
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
      email = request.form["mail"]
      password = request.form["passw"]
      try:
        auth.create_user_with_email_and_password(email, password);
        user = auth.sign_in_with_email_and_password(email, password)   
        #session
        user_id = user['idToken']
        user_email = email
        session['usr'] = user_id
        session["email"] = user_email
        return redirect("ipAddress") 
      except:
        return render_template("login.html", message="The email is already taken, try another one, please" )  
    return render_template("register.html")

@app.route("/ipAddress",methods=["GET", "POST"])
def ipAddress():
    if request.method == "POST":
        firstIpAddress  = request.form['ipAddress']
        opt  = ipCaoncatenate(firstIpAddress)
        darknet  = Darknet(opt)
        if darknet.webcam:
            dataset = LoadStreams(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)

        def detect_gen(dataset, feed_type):
            view_img = check_imshow()
            t0 = time.time()
            countFrame = 9
            dbCount = 5
            dbCount = db.child("count").get().val()
            for path, img, img0s, vid_cap in dataset:
                img = darknet.preprocess(img)
                t1 = time.time()
                pred = darknet.model(img, augment=darknet.opt["augment"])[0]  # 0.22s
                pred = pred.float()
                pred = non_max_suppression(pred, darknet.opt["conf_thres"], darknet.opt["iou_thres"])
                t2 = time.time()
                pred_boxes = []

                for i, det in enumerate(pred):
                    if darknet.webcam:  # batch_size >= 1
                        feed_type_curr, p, s, im0, frame = "Camera_%s" % str(i), path[i], '%g: ' % i, img0s[i].copy(), dataset.count
                    else:
                        feed_type_curr, p, s, im0, frame = "Camera", path, '', img0s, getattr(dataset, 'frame', 0)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {darknet.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls_id in det:
                            lbl = darknet.names[int(cls_id)]
                            xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                            score = round(conf.tolist(), 3)
                            label = "{}: {}".format(lbl, score)
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            pred_boxes.append((x1, y1, x2, y2, lbl, score))
                            if view_img:
                                darknet.plot_one_box(xyxy, im0, color=(255, 0, 0), label=label)
                    if countFrame %10 ==0:
                        detectlist  = [f'{s}']
                        for x in detectlist:
                            firstList = (x[11:])
                            chunks = firstList.split(',')
                            # presentime = datetime.now()
                            localtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for i in chunks:
                                if i != ',':
                                    print(dbCount)
                                    incidentCount = i[:1]
                                    actualData = i[2:]
                                    data = {"Detection": actualData,"Ip": firstIpAddress, 'Incident': incidentCount, 'Time':localtime }
                                    db.child("DetectionResult/").child(dbCount).set(data)
                                    dbCount = dbCount+1
                                    db.child("count").set(dbCount)
                                        
                    countFrame = countFrame + 1 
                    if feed_type_curr == feed_type:
                        frame = cv2.imencode('.jpg', im0)[1].tobytes()
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        
        @app.route('/video_feed/<feed_type>')
        def video_feed(feed_type):
            if feed_type == 'Camera_0':
                feedkBack =  Response(detect_gen(dataset=dataset, feed_type=feed_type),mimetype='multipart/x-mixed-replace; boundary=frame')
                return feedkBack
        return redirect(url_for('camIndex'))
    return render_template("IpAddress.html")

@app.route("/camIndex")
def camIndex():
    return render_template("index2.html")





@app.route("/index/socialDistanceDetection")
def socialDistanceDetection():
    return render_template('socialDistanceDetection.html')

@app.route('/social_distance_detection_video_feed')
def social_distance_detection_video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="5000")