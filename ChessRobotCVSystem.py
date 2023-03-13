import torch
import numpy as np
import cv2 as cv
# from time import time
# import requests

# Paramaters/Constants
width = 640
height = 640

# FOR PHONE TEST
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
evt = 0
def mouseClicked(event,xPos,yPos,flags,params):
    global evt
    global pnt
    if event == cv.EVENT_LBUTTONDOWN:
        print ('Mouse Event was: ', event)
        print ('At Position',xPos,yPos)
        pnt =(xPos,yPos)
        evt = event 

    if event == cv.EVENT_LBUTTONUP:
        print ('Mouse Event was: ', event)
        print ('At Position',xPos,yPos) 
        evt = event  
     

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
# url = "http://192.168.217.162:8080/shot.jpg"

     

class  piece:
    def __init__(self, x, y, name, percent):
        self.name = name
        self.x = x
        self.y = y
        self.percent = percent



class ChessPieceDetector:

    """
    Class implements Yolo5 model to detect ches peices and position on the board.
    """

    def __init__(self, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """

        
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)


    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # if model_name:
        #     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        # else:
        #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # return model
    

        #Offline Test
        path_hubconfig = 'C:/Users/johnk/OneDrive/Documents/PythonFiles/100 ChessVisionCode/yolov5-7.0'
        # path_trained_model = 'C:/Users/johnk/OneDrive/Documents/PythonFiles/100 ChessVisionCode/yolov5-7.0/models/RCBLV10.pt'
        path_trained_model = 'ChessPieceModel_Weigths/RCBLV12.pt'
        model = torch.hub.load(path_hubconfig, 'custom', path=path_trained_model, source='local')  # local repo        
        return model
    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def board_detection(self):
        """
        Finds The Corners of a chess board
        """
        # Paramaters/Constants
        chessboardSize =(7,7)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        width = 400
        height = 400

     
        #Extra Processing function For many countours in frame/ROI For largets Contour
        def ExtraPreProcess(image):
            
            qh = int(150)
            qw = int(60)

            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)   #Change to Gray Scale   
            image = cv.Canny(image, 125, 175)                 #Find only contours of image
            # cv.imshow('Canny Edges', image)
            #Find Contours
            contours, hierarchies = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            # print(f'{len(contours)} contours(s) found!')

            #Find largest Area Contour
            areas = [cv.contourArea(c) for c in contours]   #Find area of all contour
            max_index = np.argmax(areas)                    #Get Index of largest contour
            cnt=contours[max_index]                         
            x,y,w,h = cv.boundingRect(cnt)                    #Get Cordinatas of largest contour
            cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)   #Draw rectangle around largest contour
            # cv.imshow('BOUNDING BOX', image)                  #Show image with bounding box        
            ROI = image[y-qh:y+h+qh,x-qw:x+w+qw]                #create Region of Intreast to search smaller area for chess board corners
            # cv.imshow('ROI', ROI)                              #Show ROI

            gray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
            gauss = cv.GaussianBlur(gray, (3,3), 0)
            # cv.imshow('gauss', gauss)


        # Get Image from camera
        PreProcessedIMG = self.get_image()  #Process Image Before checking for chess board corners


        # Check for chess board corners
        ret, corners1 = cv.findChessboardCorners(PreProcessedIMG, chessboardSize, None)
        print(ret)

        # If Ret = ture then inner corners are found
        if ret == True:
            corners = cv.cornerSubPix(PreProcessedIMG, corners1, (1,1), (-1,-1), criteria)    # SubPix Finds more acurately measure corners

            # cv.drawChessboardCorners(img, chessboardSize, corners, ret)             #Optional draws corners on "img"
            # cv.imshow('Image', img)     

        

            # Checks if corner[0] is started at the top left of the screen or bottom right
            if (corners[0][0][1] < corners[48][0][1]):
                print('Red is on the Top Left')
            
                # Finding the outside corners For CORNER 0 being on the top Left
                #Top Left corner
                pnt0 = (corners[0][0])  
                pnt1 = (corners[1][0])
                pnt7 = (corners[7][0])
                dxTL = pnt0 - pnt1 + pnt0 - pnt7
                TopLeftCorner = pnt0 + dxTL

                #Top right corner
                pnt6 = (corners[6][0])
                pnt5 = (corners[5][0])
                pnt13 = (corners[13][0])
                dxTR = pnt6 - pnt5 + pnt6 - pnt13
                TopRightCorner = pnt6 + dxTR

                #Bottom Left corner
                pnt42 = (corners[42][0])
                pnt35 = (corners[35][0])
                pnt43 = (corners[43][0])
                dxBL = pnt42 - pnt35 + pnt42 - pnt43
                BottomLeftCorner = pnt42 + dxBL

                #Bottom right corner
                pnt48 = (corners[48][0])
                pnt47 = (corners[47][0])
                pnt41 = (corners[41][0])
                dxBR = pnt48 - pnt47 + pnt48 - pnt41
                BottomRightCorner = pnt48 + dxBR

            if (corners[0][0][1] > corners[48][0][1]):
                print('Red is on the bottom Right')
                # Finding the outside corners For CORNER 0 being on the bottome Right
                
                #Bottom Right corner
                pnt0 = (corners[0][0])  
                pnt1 = (corners[1][0])
                pnt7 = (corners[7][0])
                dxBR = pnt0 - pnt1 + pnt0 - pnt7
                BottomRightCorner = pnt0 + dxBR

                #Bottom Left corner
                pnt6 = (corners[6][0])
                pnt5 = (corners[5][0])
                pnt13 = (corners[13][0])
                dxBL = pnt6 - pnt5 + pnt6 - pnt13
                BottomLeftCorner = pnt6 + dxBL

                #Top Right corner
                pnt42 = (corners[42][0])
                pnt35 = (corners[35][0])
                pnt43 = (corners[43][0])
                dxTR = pnt42 - pnt35 + pnt42 - pnt43
                TopRightCorner = pnt42 + dxTR

                #Top Left corner
                pnt48 = (corners[48][0])
                pnt47 = (corners[47][0])
                pnt41 = (corners[41][0])
                dxTL = pnt48 - pnt47 + pnt48 - pnt41
                TopLeftCorner = pnt48 + dxTL

            #Seperate the corners cordinates into x and y and draw circles around them
            #Print TLC
            pntxTL,pntyTL = TopLeftCorner
            # pntxTL = int(pntxTL)
            # pntyTL = int(pntyTL)
            cv.circle(PreProcessedIMG,(pntxTL,pntyTL), 5, (0,0,0), thickness=-1) 

            #print TRC
            pntxTR,pntyTR = TopRightCorner
            # pntxTR = int(pntxTR)
            # pntyTR = int(pntyTR)
            cv.circle(PreProcessedIMG,(pntxTR,pntyTR), 5, (0,0,0), thickness=-1) 

            #print BLC
            pntxBL,pntyBL = BottomLeftCorner
            # pntxBL = int(pntxBL)
            # pntyBL = int(pntyBL)
            cv.circle(PreProcessedIMG,(pntxBL,pntyBL), 5, (0,0,0), thickness=-1) 

            #print BRC
            pntxBR,pntyBR = BottomRightCorner
            # pntxBR = int(pntxBR)
            # pntyBR = int(pntyBR)
            cv.circle(PreProcessedIMG,(pntxBR,pntyBR), 5, (0,0,0), thickness=-1) 

            # Perspective Warp
            pts1 = np.float32([[pntxTL,pntyTL],[pntxTR,pntyTR],[pntxBL,pntyBL],[pntxBR,pntyBR]])
            pts2 = np.float32([ [0,0], [width,0],[0,height],[width,height],])
            matrix = cv.getPerspectiveTransform(pts1,pts2)
            return matrix
            # imgOutput = cv.warpPerspective(PreProcessedIMG, matrix,(width,height))


            # DRAW Grid to show how well the board is mapped
            # GridHeight = imgOutput.shape[0] //8 
            # GridWidth = imgOutput.shape[1] // 8
            # cv.line(imgOutput,(0,0) ,(imgOutput.shape[1],0), (255,0,0), thickness=3)
            # cv.line(imgOutput,(0,0) ,(0,imgOutput.shape[0]), (255,0,0), thickness=3)
            # for i in range(8):
            #     cv.line(imgOutput,(0,i*GridHeight) ,(width,i*GridHeight), (255,0,0), thickness=3)
            # for i in range(8):
            #     cv.line(imgOutput,(i*GridWidth,0) ,(i *GridWidth,height), (255,0,0), thickness=3)
            


            #Display Images
            # cv.imshow('Image', PreProcessedIMG)
            # cv.imshow('Output', imgOutput)
            # cv.moveWindow('Image', 0,0)
            # cv.moveWindow('Output', 640,0)




        cv.waitKey(0)

    def get_image(self):
        """
        Used to take a picture from the camera
        Also does pre prossesing on the image
        """
        img1 = cv.imread('ChessBoardTest/Board_and_Pieces_Images/TestImages/ValidT1.jpg')
        # img1 = cv.imread('ChessBoardTest/Board_and_Pieces_Images/TestImages/BP6.jpg')   #Test for YOLOv5 model
        
        img = cv.resize(img1, (img1.shape[1]//6,img1.shape[0]//6), interpolation= cv.INTER_AREA)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   #Change to Gray Scale
        # img = cv.GaussianBlur(img, (3,3), 0)        #Blur Image


        # img = cv.imread('ChessBoardTest/Board_and_Pieces_Images/2_15_2023_Test_Images/WS2.jpg') #Test Image
        # Test Code for IP Web camera
        # img_resp = requests.get(url)
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # img = cv.imdecode(img_arr, -1)
        # img = cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
        return img

    def get_matrix(self, frame):
        # Manual Input of corners for perspective warp
        #Print TLC
        pntxTL = int(130)
        pntyTL = int(138)
        cv.circle(frame,(pntxTL,pntyTL), 2, (255,0,0), thickness=-1) 
        #print TRC
        pntxTR = int(295)
        pntyTR = int(137)
        cv.circle(frame,(pntxTR,pntyTR), 2, (0,255,0), thickness=-1) 
        #print BLC
        pntxBL = int(90)
        pntyBL = int(346)
        cv.circle(frame,(pntxBL,pntyBL), 2, (0,255,0), thickness=-1) 
        #print BRC
        pntxBR = int(342)
        pntyBR = int(342)
        cv.circle(frame,(pntxBR,pntyBR), 2, (255,0,0), thickness=-1)

        # Perspective Warp
        pts1 = np.float32([[pntxTL,pntyTL],[pntxTR,pntyTR],[pntxBL,pntyBL],[pntxBR,pntyBR]])
        pts2 = np.float32([ [0,0], [width,0],[0,height],[width,height],])
        matrix = cv.getPerspectiveTransform(pts1,pts2)
        return matrix

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.4:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                yDisplacement = int(10)
                newx = int((x1+ x2)/2) 
                newy = y2 - yDisplacement
                
                bgr = (255, 0, 0)
                cv.circle(frame, (newx,newy), 3, bgr,thickness= -1)

                cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv.putText(frame, self.class_to_label(labels[i]), (newx,newy), cv.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 1)

        return frame
    
    def warp_points(self, results, frame, matrix):
        """
        Takes cordinates from results and warpes them into cordinate system of just the board

        """
        labels, cord = results
        detectedPieces = []

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.4:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                yDisplacement = int(10)
                newx = int((x1+ x2)/2) 
                newy = y2 - yDisplacement

                OriginalPoint = np.single([[[newx,newy]]])
                xO = int(OriginalPoint[0][0][0])
                yO =  int(OriginalPoint[0][0][1])
                cv.circle(frame,(xO,yO), 5, (0,0,255), thickness= 1) 

                # Save new (x,y) cordinates back in the row
                # Transfers original points onto the new perspective warped image
                Outputpoints = cv.perspectiveTransform(OriginalPoint, matrix)
                # print('Output Points')
                # print(Outputpoints)
                xN = int(Outputpoints[0][0][0])
                yN =  int(Outputpoints[0][0][1])
                detectedPieces.append(piece(xN,yN,self.class_to_label(labels[i]),round(row[4].item() ,3)))

                # print(detectedPieces[i].name)


        return detectedPieces

    def map_pieces(self, detectedPieces):
        RANK = 8
        FILES = 8
        board = [['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.'], 
                ['.', '.', '.', '.', '.', '.', '.', '.']]
        n = len(detectedPieces)
        # print(n)
        for i in range(n):
            print('(' , detectedPieces[i].x, ',', detectedPieces[i].y, ') ' , detectedPieces[i].name, '/', detectedPieces[i].percent)

        #     # Map Files
            if(detectedPieces[i].x >= 0 and detectedPieces[i].x < width/8):
                file = int(0)
            elif(detectedPieces[i].x >=  width/8 and detectedPieces[i].x < width*2/8):
                file = int(1)
            elif(detectedPieces[i].x >=  width*2/8 and detectedPieces[i].x < width*3/8):
                file = int(2)
            elif(detectedPieces[i].x >=  width*3/8 and detectedPieces[i].x < width*4/8):
                file = int(3)
            elif(detectedPieces[i].x >=  width*4/8 and detectedPieces[i].x < width*5/8):
                file = int(4)
            elif(detectedPieces[i].x >=  width*5/8 and detectedPieces[i].x < width*6/8):
                file = int(5)
            elif(detectedPieces[i].x >=  width*6/8 and detectedPieces[i].x < width*7/8):
                file = int(6)
            elif(detectedPieces[i].x >=  width*7/8 and detectedPieces[i].x < width*8/8):
                file = int(7)
            
            # Map Rank
            if(detectedPieces[i].y >= 0 and detectedPieces[i].y < width/8):
                rank = int(0)
            elif(detectedPieces[i].y >=  height/8 and detectedPieces[i].y < height*2/8):
                rank = int(1)
            elif(detectedPieces[i].y >=  height*2/8 and detectedPieces[i].y < height*3/8):
                rank = int(2)
            elif(detectedPieces[i].y >=  height*3/8 and detectedPieces[i].y < height*4/8):
                rank = int(3)
            elif(detectedPieces[i].y >=  height*4/8 and detectedPieces[i].y < height*5/8):
                rank = int(4)
            elif(detectedPieces[i].y >=  height*5/8 and detectedPieces[i].y < height*6/8):
                rank = int(5)
            elif(detectedPieces[i].y >=  height*6/8 and detectedPieces[i].y < height*7/8):
                rank = int(6)
            elif(detectedPieces[i].y >=  height*7/8 and detectedPieces[i].y < height*8/8):
                rank = int(7)
            print(file, rank, '\n')
    
            board[rank][file] = detectedPieces[i].name
        # Prints Board in a matrix array
        print(board[0],'\n',board[1],'\n',board[2],'\n',board[3],'\n',board[4],'\n',board[5],'\n',board[6],'\n',board[7],'\n')


    def plot_new_pieces(self, detectedPieces, imgOutput):
        n = len(detectedPieces)
        for i in range(n):
            cv.circle(imgOutput,(detectedPieces[i].x,detectedPieces[i].y), 5, (255,255,255), thickness= -1) 
        return imgOutput

    
    def __call__(self):
        """
        This Function does the main Peice detection
        When called the following steps are taken
        1) Get image 
            -Call get_image() /resize to 416X416 and gray scale/
        2) Detect Pieces
            -Run self.score_frame(frame)
        3) Warp Piece locations
            -Run self.warp_points(results, frame, PerspectiveMatrix )
        4) Map pieces
            -Run self.map_pieces(detectedPieces)
        5) Return Matrix array of board
            - return Matrix

        """
        




        # # img = cv.imread('ChessBoardTest/Board_and_Pieces_Images/TestImages/BP6.jpg')   #Image with corner points added added
        # img = cv.imread('ChessBoardTest/Board_and_Pieces_Images/2_15_2023_Test_Images/WS2.jpg') #Test Image

        # # cam = cv.VideoCapture(0, cv.CAP_DSHOW)
        # # img = cam.read()
        img = self.get_image()
        # img = cv.resize(img, (416,416), interpolation=cv.INTER_AREA)
        frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        
        results = self.score_frame(frame)
        frame = self.plot_boxes(results, img)      
     
        # # matrix = self.get_matrix(frame)
        # # imgOutput = cv.warpPerspective(img, matrix,(width,height))

        # # Warp point to new Image
        # detectedPieces = self.warp_points(results, frame, matrix )
        
        # self.map_pieces(detectedPieces)
        # imgOutput = self.plot_new_pieces(detectedPieces, imgOutput)



        # cv.imshow('YOLOv5 Detection', frame)
        # cv.moveWindow('YOLOv5 Detection', 0,0)
        # cv.imshow('imgOutput', imgOutput)
        # # cv.moveWindow('imgOutput', 416,0)
        # cv.waitKey(0)
        # return matrix

        cv.imshow('Android_cam', frame)
        cv.waitKey(0)

        

# w = input('Press Enter to SETUP Detector')
detector = ChessPieceDetector(model_name='ChessPieceModel_Weigths/RCBLV11.pt')

  



# x = input('Press Enter to send poitns from Detector')
# A = detector.board_detection()
# print(A)

# RUN TO TEST WORKING
# w = input('Press Enter to Run Detector')
detector()




# Test Phone Web link
# foundmatrix = 0
# cv.namedWindow('Android_cam')
# cv.setMouseCallback('Android_cam',mouseClicked)
# while True:


#     if evt == 1:
#         # if (foundmatrix == 0):
#         #     matrix = detector.board_detection()
#         #     foundmatrix = 1

#         # elif(foundmatrix == 1):
#         detector()

#         #cv2.circle(img,pnt, 25,(255,0,0), 2)
  
#     # Press Esc key to exit
#     if cv.waitKey(1) == 27:
#         break
  
# cv.destroyAllWindows()








