import pygame
from pygame.locals import *
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import operator
import math
import time
import random

def text_to_screen(screen, text, x, y, size ,
            color,font_type ):
    try:
        
        text = str(text)
        font = pygame.font.Font(font_type, size)
        
        text = font.render(text, True, color)
        screen.blit(text, (x, y))

    except Exception as e:
        print ('eeee ',e)
def Reverse(lst): 
    return [ele for ele in reversed(lst)] 

def addBall(balls,x,y,xs,ys,s,r,g,b):
    balls.append({"x":x,"y":y,"xs":xs,"ys":ys,"size":s,"r":r,"g":g,"b":b})

def addPlatform(platform,x1,y1,x2,y2):
    dx,dy,dist=getV(x1,y1,x2,y2)
    dx,dy=normalise(dx,dy,dist)
    platform.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"tx":dx,"ty":dy,"nx":(0-dy),"ny":dx,"length":dist})
    
def init(balls,platform):
    balls=balls
    platform=[]
    
def getV(x1,y1,x2,y2): #get length,xchange,ychange
    dx=(x2-x1)
    dy=(y2-y1)
    dist = ((dx*dx)+(dy*dy))
    dist = math.sqrt(dist)
    return (dx,dy,dist)

def normalise(dx,dy,dist): #normalise dx and dy
    if not(dist==0):
        dx2=(dx/dist)
        dy2=(dy/dist)
        return (dx2,dy2)

def ballchange(balls,index,i2,val):   #used to change one value easily
    balls[index][i2] = (balls[index][i2]+val)
    
def deleteball(balls,index): #deletes a ball
    balls.pop(index)
      
def checkBC(balls,bi1,bi2,bounce): #collision between two balls
    dx,dy,dist = getV(balls[bi1]["x"],balls[bi1]["y"],balls[bi2]["x"],balls[bi2]["y"]) #get dx,dy,length
    depth = ((balls[bi1]["size"]+balls[bi2]["size"])-dist) #calculate how much the forst ball has gone into the second one
    if depth > 0: #if they are colliding
        dx,dy = normalise(dx,dy,dist) #normalise dx and dy
        depth = depth/2
        ballchange(balls,bi1,"x",(dx*(0-depth))) #move the first ball out of the second
        ballchange(balls,bi1,"y",(dy*(0-depth))) 
        ballchange(balls,bi2,"x",(dx*depth)) #move the second ball out of the first
        ballchange(balls,bi2,"y",(dy*depth))
        rvx = (balls[bi2]["xs"]-balls[bi1]["xs"]) #difference of x positions of both the balls
        rvy = (balls[bi2]["ys"]-balls[bi1]["ys"]) #difference of y positions of both the balls
        rv = (dx*rvx)+(dy*rvy)
        rv = (-1-bounce)*(rv/2) #calculate bounce
        ballchange(balls,bi1,"xs",(dx*(0-rv))) #set the velocities using rv. This piece will make the ball bounce
        ballchange(balls,bi1,"ys",(dy*(0-rv)))
        ballchange(balls,bi2,"xs",(dx*(rv)))
        ballchange(balls,bi2,"ys",(dy*(rv)))
    
def colBall(balls,index,pos,pRad,bounce): #the ball-platform collision physics
    #just like the ball to ball collision. This time, only the ball moves and not the platform.
    dx,dy,dist=getV(pos[0],pos[1],balls[index]["x"],balls[index]["y"])
    depth = ((pRad+balls[index]["size"])-dist) #calculate how much the ball has got into the platform
    if depth > 0: #if they collide
        dx,dy=normalise(dx,dy,dist) #normalise
        if dy < 0: #if it has got too much into the platform, it will mess up the bounce effect
            depth = (depth+0.1) #push it out of the platform a bit.
        ballchange(balls,index,"x",(dx*depth)) #get the ball out of the platform
        ballchange(balls,index,"y",(dy*depth))
        velop = ((dx*balls[index]["xs"])+(dy*balls[index]["ys"])) #set the velocity based on the ball velocity and how much it has got into the platform
        velop = ((0-bounce)*velop) #add the bounce effect
        ballchange(balls,index,"xs",(dx*velop)) #change the ball velovity using it's bounce(the velop variable)
        ballchange(balls,index,"ys",(dy*velop))

def checkcol(balls,platform,bi,pi,pRad,bounce): #checks if a ball is colliding with a platform
    dx,dy,dist=getV(platform[pi]["x1"],platform[pi]["y1"],balls[bi]["x"],balls[bi]["y"])
    tp=((dx*platform[pi]["tx"])+(dy*platform[pi]["ty"])) #project
    if tp < 0:
        tp = 0
    if tp > platform[pi]["length"]:
        tp=platform[pi]["length"]
    colBall(balls,bi,[(platform[pi]["x1"]+(tp*platform[pi]["tx"])),(platform[pi]["y1"]+(tp*platform[pi]["ty"]))],pRad,bounce)
    
def update(balls,platform,pRad,bounce,gravity,friction,bbounce,w,h): #the main function
    bi=0
    for bi in range(len(balls)): #change the ball position according to it's velocity,gravity and friction
        if bi < len(balls):
            ballchange(balls,bi,"x",balls[bi]["xs"])
            ballchange(balls,bi,"y",balls[bi]["ys"])
            ballchange(balls,bi,"xs",0)
            ballchange(balls,bi,"ys",gravity)
            balls[bi]["xs"] = balls[bi]["xs"]*friction
            for ob in range(len(balls)): #check ball to ball collision
                if not(bi == ob): #avoid checking collission with the selected ball itself
                    checkBC(balls,bi,ob,bbounce)        
            pi=0
            for pi in range(len(platform)): #check collision with platform
                checkcol(balls,platform,bi,pi,pRad,bounce)
            if balls[bi]["x"]>w:
                deleteball(balls,bi)
            elif balls[bi]["y"]>h:
                deleteball(balls,bi)

def anToXY(angle,rad,x,y): #There is no use for this one. I made it to draw the ball angle
    x = x+(rad*(math.cos(angle)))
    y = y+(rad*(math.sin(angle)))
    return (x,y)

def drawLine(x1,y1,x2,y2,size,col):
    pygame.draw.line(win, (0,0,0), [x1,y1], [x2,y2], size+1)
    pygame.draw.circle(win, col, (x1, y1), size, size)   
    pygame.draw.circle(win, col, (x2, y2), size, size)   
    
def render(balls,platform,win,pCol,pRad): #this function does all the drawing
    #drawing
    pi=0
    for pi in range(len(platform)): #draws the platforms
        pygame.draw.line(win, pCol, [platform[pi]["x1"],platform[pi]["y1"]], [platform[pi]["x2"],platform[pi]["y2"]], (pRad*2))
        pygame.draw.circle(win, pCol, (int(platform[pi]["x1"]), int(platform[pi]["y1"])), pRad, pRad-1)   
        pygame.draw.circle(win, pCol, (int(platform[pi]["x2"]), int(platform[pi]["y2"])), pRad, pRad-1)

        cv2.line(frame,(platform[pi]["x1"],platform[pi]["y1"]),(platform[pi]["x2"],platform[pi]["y2"]),(0,255,255),10)
        cv2.circle(frame, (int(platform[pi]["x1"]), int(platform[pi]["y1"])), pRad, -1)   
        cv2.circle(frame, (int(platform[pi]["x2"]), int(platform[pi]["y2"])), pRad, -1)      
        
    pi=0
    for pi in range(len(balls)): #draw the balls
        pygame.draw.circle(win, (balls[pi]["r"],balls[pi]["g"],balls[pi]["b"]), (int(balls[pi]["x"]), int(balls[pi]["y"])), balls[pi]["size"], balls[pi]["size"])   
        pygame.draw.circle(win, (0,0,0), (int(balls[pi]["x"]), int(balls[pi]["y"])), balls[pi]["size"], 1) 

        cv2.circle(frame, (int(balls[pi]["x"]), int(balls[pi]["y"])), balls[pi]["size"],(balls[pi]["r"],balls[pi]["g"],balls[pi]["b"]), -1)   
        cv2.circle(frame, (int(balls[pi]["x"]), int(balls[pi]["y"])),balls[pi]["size"],(balls[pi]["r"],balls[pi]["g"],balls[pi]["b"]), -1)   

def setting1(platform,balls): #you can change this. Just the platforms and balls

    addBall(balls,1115,-250,0,2,18,255,150,255) #light pink
    addBall(balls,1100,-300,0,0,9,0,255,0) #light green

    
def setting2(platform,balls): #another setting I made
    addPlatform(platform,1200,30,200,130)


def setting3(balls,platform):
    #custom
    print("nothing here")
    
def physics(balls,platform,pRad,bounce,gravity,fric,bbounce,win,pCol,w,h):
    render(balls,platform,win,pCol,pRad)
    update(balls,platform,pRad,bounce,gravity,fric,bbounce,w,h)

def create_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(18, activation='softmax'))
    return model
def gcd(x, y):
    while y != 0:
        (x, y) = (y, x % y)
    #print(x)
    return(x)

model = create_model()
model.load_weights('/home/pragyanshu/work/6_aruco_markers/Aruco_Tracker-master/onboard/1_quad_factoring/model_mnist5.h5')


cap = cv2.VideoCapture(0)
pygame.init()


####
gravity = 0.9
bounce = 1.4
bbounce = 0.9
pRad = 10
fric = 1
pCol = (79,179,255)
#width and hieght of the window. You can change this
wind_w = 1200 #the maximum width and hieght I can use in my monitor without covering the dock 
wind_h = 650
#the main code
balls = []
platform = []
init(balls,platform) #clear everything
setting2(platform,balls)
size = (wind_w, wind_h)
pygame.display.set_caption("OpenCV camera stream on Pygame")
win = pygame.display.set_mode([wind_w,wind_h])
clock = pygame.time.Clock()
done = True
addp = "n"

st = ""
s=""
no_of_times = 0
No=0

#try:
while True:

        ret, frame = cap.read()

        frame = cv2.resize(frame,(wind_w,wind_h))

        ##################### 
        frame1 = frame.copy()
        frame_new = frame.copy()
        #finding contours
        ret, img = cv2.threshold(frame1, 78, 255, cv2.THRESH_BINARY_INV)
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgnew, contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        thisdict = {}
        
        flag=0
        noted_y=0
        mylist = []
        new_x_cor=[]
        x_cor_numb = []
        for c in contours:
            (x, y, w, h)= cv2.boundingRect(c)
            if (w>40) or (h>40):
                
                mylist.append((x,y,w,h))
                #x_cor_numb.append(x)
                for i in range(0, len(mylist)):
                    x = mylist[i][0]
                    y = mylist[i][1]
                    w = mylist[i][2]
                    h = mylist[i][3]
                    if h/w>3:
                        x=x-20
                        w=w+50
                    if w/h>3:
                        y=y-40
                        h=h+130
                    y=y-30
                    x=x-30
                    w=w+60
                    h=h+60
                    cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0, 255), 2)
                    img1 = frame_new[y:y+h, x:x+w]
                    ret, gray = cv2.threshold(img1,108,255,cv2.THRESH_BINARY )
                    try:
                        im = cv2.resize(gray, (28,28))
                        ar = np.array(im).reshape((28,28,3))
                        ar = np.expand_dims(ar, axis=0)
                        prediction = model.predict(ar)[0]

                        #predicrion of class labels
                        for i in range(0,19):
                            if prediction[i]==1.0:
                                if i==0:
                                    j= "+"
                                if i==1:
                                    j= "-"
                                if i==2:
                                    j= "0"
                                if i==3:
                                    j= "1"
                                if i==4:
                                    j= "2"
                                if i==5:
                                    j= "3"
                                if i==6:
                                    j= "4"
                                if i==7:
                                    j= "5"
                                if i==8:
                                    j= "6"
                                if i==9:
                                    j= "7"
                                if i==10:
                                    j= "8"
                                if i==11:
                                    j= "9"
                                if i==12:
                                    j= "="
                                if i==13:
                                    j= "^"
                                if i==14:
                                    j= "/"
                                if i==15:
                                    j= "X"
                                if i==16:
                                    j= "x"
                                if i==17:
                                    j= "y"  
                        #printing prediction
                                cv2.putText(frame, j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                                thisdict[x]= str(j)
                                if j in ["0","1","2","3","4","5","6","7","8","9"]:
                                    x_cor_numb.append(x+30)
                                    #break
                    except:
                        d=0
           
            sort = sorted(thisdict.items(), key=operator.itemgetter(0))
            s = ""
            
            for x in range(0,len(sort)):
                if x not in ['^','X','x','y','=','-','/']:
                    s=s+str(sort[x][1])
        
        #cv2.putText(frame, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        numbs = []
        
       # print(s,'00000000000000000000',x_cor_numb)
        splited_s = s.split('+')
        xc=50;yc=100

        t=0
        for i in splited_s:
            n= len(i)
            if n>0:
                summ=0
                a = x_cor_numb[t:t+n]
                summ = int(sum(a)/n)
                t = t+n
                new_x_cor.append(summ)
        print('new coor ',splited_s)
        for i in new_x_cor:
                cv2.putText(frame, 'S', (i,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)


        try:
            for i in splited_s:
                numbs.append(int(i))

        except:
            print('eee')
        #print('numbs ',numbs) 

        list_cor=[(10,wind_h-10,wind_w,wind_h-10)]
        #list_cor=None
        #--------------
        x,y = pygame.mouse.get_pos()
        win.fill([0,0,0])

        if list_cor is not None:
                #print(list_cor,' setttttt')
                balls=balls;platform=[]
                init(balls,platform)
                for a in list_cor:
                    #print(a)
                    
                    addPlatform(platform,a[0],a[1],a[2],a[3])
        elif list_cor is None:
                balls=balls;platform=[]
                init(balls,platform)

        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                done = False

            if event.type == pygame.KEYDOWN: #user interface-sensing mouse clicks for adding balls
                if event.key == pygame.K_LCTRL:
                    x,y = pygame.mouse.get_pos()
                    if addp == "n": #If pressing ctrl for the first time, start a line
                        x1 = x
                        y1 = y
                        addp = "y"
                    if not(x1 == x) or not(y1 == y):
                        if addp == "y": #if pressing for the scond time, add a platform
                            pygame.draw.line(win, pCol, (x1, y1), (x, y),pRad*2)
                            if not(x1 == x) or not(y1 == y):
                                addp = "n"
                                addPlatform(platform,x1,y1,x,y)                              
                elif event.key == pygame.K_SPACE: #Delete a ball when space key is pressed and the mouse is touching that ball
                    x,y = pygame.mouse.get_pos()
                    looping = True
                    i = 0
                    if len(balls) > 0:
                        while ((looping and i <= len(balls))): #repeat until each ball is checked or mouse is touching a ball
                            dx,dy,dist = getV(x,y,balls[i]["x"],balls[i]["y"])
                            if dist <= balls[i]["size"]:
                                deleteball(balls,i)
                                looping = False
                            i+=1
                            if i == len(balls):
                                    break

            elif event.type == pygame.MOUSEBUTTONDOWN: #If mouse clicked, add a ball
                x,y = pygame.mouse.get_pos() 
                
                xx1=x
                try:
                    x_cor_numb = list(dict.fromkeys(x_cor_numb))
                    print('befor ',x_cor_numb)
                    x_cor_numb = Reverse(x_cor_numb)
                    print('after ',x_cor_numb)

                    j=-1; yy1=y
                    if len(numbs) >0:
                        for No in numbs:
                            j=j+1
                            xx1 = x_cor_numb[j]
                            b,g,r = random.randint(0,255),random.randint(0,255),random.randint(0,255)
                            
                            i=1
                            while True:
                                addBall(balls,xx1,yy1,0,0,10,b,g,r)               
                                xx1=xx1+10
                                if i == No:
                                    break
                                i=i+1
                except:
                    xx1=x;yy1=y
                    print('error in new_x_cor')


                dx,dy,dist = getV(xx1,yy1,15,15) #Sensing the reset button
                if int(dist) < 15:
                    balls = []
                    platform = []

        if addp == "y": #draw the temporary line from mouse to start of the line
            pygame.draw.line(win, pCol, (x1, y1), (x, y),pRad*2)
            pygame.draw.circle(win, pCol,(x1, y1),pRad, pRad)  
            pygame.draw.circle(win, pCol,( x, y,), pRad, pRad)  

        physics(balls,platform,pRad,bounce,gravity,fric,bbounce,win,pCol,wind_w,wind_h) #This function contains the physics engine
        pygame.draw.circle(win, (255,150,0),(15, 15),15, 15)  #the reset button
        #pygame.display.flip()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = cv2.flip(frame,0)
        frame = pygame.surfarray.make_surface(frame)
        win.blit(frame, (0,0))
        pygame.display.update()
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#except Exception as e:
#    print('ERROR:',e)
#    pygame.quit()
#    cv2.destroyAllWindows()
        ###########################################
    
cap.release()
cv2.destroyAllWindows() 