import torch
import numpy as np
import cv2 as cv
import requests

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
url = "http://192.168.218.25:8080/shot.jpg"

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
    # Class Variables
    global perspectiveMatrix

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
    
        #Offline Test
        path_hubconfig = 'C:/Users/johnk/OneDrive/Documents/PythonFiles/100 ChessVisionCode/yolov5-7.0'
        path_trained_model = 'ChessPieceModel_Weigths/004.pt'
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
        width = 640
        height = 640

     
        #Extra Processing function For many countours in frame/ROI For largets Contour
        def ExtraPreProcess(Originalimage):
            
            qh = int(50)
            qw = int(50)

            # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)   #Change to Gray Scale   
            image = cv.Canny(Originalimage, 125, 175)                 #Find only contours of image
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
            cv.imshow('BOUNDING BOX', image)                  #Show image with bounding box        
            cv.waitKey(0)

            # # Now thar we have the largest Contour(Should be around the chess board)
            # # We will black out around the contour so the board detection will work

            # Originalimage[0:y-qh , 0:w] = (0)
            # Originalimage[y+h+qh:height,0:w] = (0)
            # Originalimage[0:height,0:x - qw] = (0)
            # Originalimage[0:height,x  +w +qw] = (0)
            # cv.imshow('Originalimage With Blacked OUT', Originalimage)   # Show Originalimage blacked out
            # return Originalimage
            
            # ROI = Originalimage[y-qh:y+h+qh,x-qw:x+w+qw]                #create Region of Intreast to search smaller area for chess board corners
            # cv.imshow('ROI', ROI)                              #Show ROI
            # return ROI
            # gray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
            # gauss = cv.GaussianBlur(ROI, (3,3), 0)
            # cv.imshow('gauss', gauss)
            # cv.waitKey(0)



        # Get Image from camera
        PreProcessedIMG = self.get_image()  #Process Image Before checking for chess board corners
        PreProcessedIMG = cv.cvtColor(PreProcessedIMG, cv.COLOR_BGR2GRAY)   #Change to Gray Scale
        PreProcessedIMG = cv.GaussianBlur(PreProcessedIMG, (3,3), 0)        #Blur Image
        cv.imshow('PreProcessedIMG', PreProcessedIMG)
        cv.waitKey(0)
        


        # Check for chess board corners
        ret, corners1 = cv.findChessboardCorners(PreProcessedIMG, chessboardSize, None)
        print(ret)
        # if ret == False:
        #     PreProcessedIMG = ExtraPreProcess(PreProcessedIMG)

        # Secound test After extra processing
        ret, corners1 = cv.findChessboardCorners(PreProcessedIMG, chessboardSize, None)
        print(ret)
        # If Ret = ture then inner corners are found
        if ret == True:
            corners = cv.cornerSubPix(PreProcessedIMG, corners1, (1,1), (-1,-1), criteria)    # SubPix Finds more acurately measure corners

            cv.drawChessboardCorners(PreProcessedIMG, chessboardSize, corners, ret)             #Optional draws corners on "img"
            # cv.imshow('Image', img)     

            corner = [0,0,0,0]

          
            # Finding the outside corners based off of the inside corners
            #corner 0
            pnt0 = (corners[0][0])  
            pnt1 = (corners[1][0])
            pnt7 = (corners[7][0])
            dxTL = pnt0 - pnt1 + pnt0 - pnt7
            corner[0] = pnt0 + dxTL

            #corner 1
            pnt6 = (corners[6][0])
            pnt5 = (corners[5][0])
            pnt13 = (corners[13][0])
            dxTR = pnt6 - pnt5 + pnt6 - pnt13
            corner[1] = pnt6 + dxTR

            #corner 2
            pnt42 = (corners[42][0])
            pnt35 = (corners[35][0])
            pnt43 = (corners[43][0])
            dxBL = pnt42 - pnt35 + pnt42 - pnt43
            corner[2] = pnt42 + dxBL

            #corner 3
            pnt48 = (corners[48][0])
            pnt47 = (corners[47][0])
            pnt41 = (corners[41][0])
            dxBR = pnt48 - pnt47 + pnt48 - pnt41
            corner[3] = pnt48 + dxBR

            # # With all the corners found. Now they need to be oriented correctly. 
            # To do this the Y values are sorted.
            # The bottom two corners having a larger Y values and the Top two have lower Y values
            # Once the Y values are sorted we chack the X value respectivly
            # The higher X value will be the right point and the Lowe X value the Left point.

            corner.sort(key=lambda a: a[1]) #Sorts the corner list by the secound element in the tupil(the Y values)
            # The corners List is now in assending order with respect to the y values
            # corner = [(x0,Y0),(x1,Y1),(x2,Y2),(x3,Y3)] # Y values are in assedning Order

            if corner[0][0]> corner[1][0]:  #If x0 > x1
                TopLeftCorner = corner[1]   # Then (x1,Y1) = TopLeftCorner
                TopRightCorner = corner[0]  # (x0,Y0) = TopRightCorner
            else:
                TopLeftCorner = corner[0]
                TopRightCorner = corner[1]


            if corner[2][0]> corner[3][0]:
                BottomRightCorner = corner[2]
                BottomLeftCorner = corner[3]
            else:
                BottomRightCorner = corner[3]
                BottomLeftCorner = corner[2]


            #Seperate the corners cordinates into x and y and draw circles around them
            #Print TLC
            pntxTL,pntyTL = TopLeftCorner
            pntxTL = int(pntxTL)
            pntyTL = int(pntyTL)
            cv.circle(PreProcessedIMG,(pntxTL,pntyTL), 5, (255,0,0), thickness=-1) 

            #print TRC
            pntxTR,pntyTR = TopRightCorner
            pntxTR = int(pntxTR)
            pntyTR = int(pntyTR)
            cv.circle(PreProcessedIMG,(pntxTR,pntyTR), 5, (0,0,0), thickness=-1) 

            #print BLC
            pntxBL,pntyBL = BottomLeftCorner
            pntxBL = int(pntxBL)
            pntyBL = int(pntyBL)
            cv.circle(PreProcessedIMG,(pntxBL,pntyBL), 5, (0,0,0), thickness=-1) 

            #print BRC
            pntxBR,pntyBR = BottomRightCorner
            pntxBR = int(pntxBR)
            pntyBR = int(pntyBR)
            cv.circle(PreProcessedIMG,(pntxBR,pntyBR), 5, (0,0,0), thickness=-1) 

            # Perspective Warp
            pts1 = np.float32([[pntxTL,pntyTL],[pntxTR,pntyTR],[pntxBL,pntyBL],[pntxBR,pntyBR]])
            pts2 = np.float32([ [0,0], [width,0],[0,height],[width,height],])
            warpMatrix = cv.getPerspectiveTransform(pts1,pts2)
            cv.imshow('Image', PreProcessedIMG)
            cv.waitKey(0)
            return warpMatrix, ret
            # imgOutput = cv.warpPerspective(PreProcessedIMG, perspectiveMatrix,(width,height))


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
        else:
            return ret, ret




        # cv.waitKey(0)

    def get_image(self):
        """
        Used to take a picture from the camera
        Also does pre prossesing on the image
        """
        # img1 = cv.imread('ChessBoardTest/Board_and_Pieces_Images/BP1.jpg')
        img1 = cv.imread('Image_Folder_000_TEST/image2.jpg')   #Test for YOLOv5 model
        

        # img = cv.resize(img1, (img1.shape[1]//6,img1.shape[0]//6), interpolation= cv.INTER_AREA)
        # img = cv.resize(img1, (416,416), interpolation=cv.INTER_AREA)
        img = cv.resize(img1, (640,640), interpolation=cv.INTER_AREA)


        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   #Change to Gray Scale
        # img = cv.GaussianBlur(img, (3,3), 0)        #Blur Image


        # Test Code for IP Web camera
        # img_resp = requests.get(url)
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # img = cv.imdecode(img_arr, -1)
        # img = cv.resize(img, (640,640), interpolation=cv.INTER_AREA)

        # Black out the outside of the board
        # x =250
        # y =185
        # w =355
        # h =405

        # img[0:y , 0:width] = (0,0,0)
        # img[y+h:height,0:width] = (0,0,0)
        # img[0:height,0:x] = (0,0,0)
        # img[0:height,x + w:width ] = (0,0,0)
        return img

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
                cv.putText(frame, self.class_to_label(labels[i]), (newx,newy), cv.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)

        return frame

    def warp_points(self, results, frame, warpMatrix):
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
                Outputpoints = cv.perspectiveTransform(OriginalPoint, warpMatrix)
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
            else:
                continue
            
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
            else:
                continue
            # print(file, rank, '\n')
    
            board[rank][file] = detectedPieces[i].name
        # Prints Board in a matrix array
        # print(board[0],'\n',board[1],'\n',board[2],'\n',board[3],'\n',board[4],'\n',board[5],'\n',board[6],'\n',board[7],'\n')
        return board
    def plot_new_pieces(self, detectedPieces, imgOutput):
        n = len(detectedPieces)
        for i in range(n):
            cv.circle(imgOutput,(detectedPieces[i].x,detectedPieces[i].y), 5, (255,255,255), thickness= -1) 
        return imgOutput

    # def __call__(self, warpMatrix):
    def __call__(self,img):

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
        
        # img = self.get_image()
        # frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        frame = img
        # AHE
        # Create a CLAHE object (Arguments are optional)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # # Apply adaptive histogram equalization
        # frame = clahe.apply(frame)

        results = self.score_frame(frame)
        print(results)
        frame = self.plot_boxes(results, img)      

        return frame

        # imgOutput = cv.warpPerspective(img, warpMatrix,(width,height))

        # # Warp point to new Image
        # detectedPieces = self.warp_points(results, frame, warpMatrix )
        
        # Board = self.map_pieces(detectedPieces)


        # imgOutput = self.plot_new_pieces(detectedPieces, imgOutput)
        
        # cv.imshow('imgOutput', imgOutput)
        # cv.waitKey(0)
        # return Board


        # cv.imshow('YOLOv5 Detection', frame)
        # cv.moveWindow('YOLOv5 Detection', 0,0)
        # # cv.imshow('imgOutput', imgOutput)
        # # # cv.moveWindow('imgOutput', 416,0)
        # cv.waitKey(0)

        # frame = cv.resize(frame, (900,700), interpolation=cv.INTER_AREA)
        # cv.imshow('Android_cam', frame)
        # cv.waitKey(0)

        


# w = input('Press Enter to SETUP Detector')
detector = ChessPieceDetector(model_name='ChessPieceModel_Weigths/RCBLV17.pt')

  



# x = input('Press Enter to send poitns from Detector')
# A = detector.board_detection()
# print(A)

# RUN TO TEST WORKING
# w = input('Press Enter to Run Detector')
# detector()




# # Test Phone Web link
# foundmatrix = 1
# # cv.namedWindow('Android_cam')
# # cv.setMouseCallback('Android_cam',mouseClicked)


# while True:


#     w = input('Press Enter to Run Detector')
#     if foundmatrix == 0:
#         warpMatrix, ret = detector.board_detection()
#         if ret == True:
#             foundmatrix = 1
#     else:
#         detector()
#         # board = detector(warpMatrix)
#         # print(board[0],'\n',board[1],'\n',board[2],'\n',board[3],'\n',board[4],'\n',board[5],'\n',board[6],'\n',board[7],'\n')

#     if w == '1':
#         break

    # if evt == 1:
    #     detector.board_detection()
    #     # if (foundmatrix == 0):
    #     #     matrix = detector.board_detection()
    #     #     foundmatrix = 1

    #     # elif(foundmatrix == 1):
        

    #     #cv2.circle(img,pnt, 25,(255,0,0), 2)
  
    # # Press Esc key to exit
    # if cv.waitKey(1) == 27:
    #     break
  



img = cv.imread('Image_Folder_000_TEST/image0.jpg')   #Test for YOLOv5 model
img0 = cv.resize(img, (640,640), interpolation=cv.INTER_AREA)

gray_img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

# Apply adaptive histogram equalization using CLAHE
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# AHE_IMAGE = clahe.apply(gray_img)
frame0 = detector(gray_img)

# img = cv.imread('Image_Folder_000_TEST/image1.jpg')   #Test for YOLOv5 model
# img1= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame1 = detector(img1)

# img = cv.imread('Image_Folder_000_TEST/image2.jpg')   #Test for YOLOv5 model
# img2= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame2 = detector(img2)

# img = cv.imread('Image_Folder_000_TEST/image3.jpg')   #Test for YOLOv5 model
# img3= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame3 = detector(img3)

# img = cv.imread('Image_Folder_000_TEST/image4.jpg')   #Test for YOLOv5 model
# img4= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame4 = detector(img4)

# img = cv.imread('Image_Folder_000_TEST/image5.jpg')   #Test for YOLOv5 model
# img5= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame5 = detector(img5)

# img = cv.imread('Image_Folder_000_TEST/image6.jpg')   #Test for YOLOv5 model
# img6= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame6 = detector(img6)

# img = cv.imread('Image_Folder_000_TEST/image7.jpg')   #Test for YOLOv5 model
# img7= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame7 = detector(img7)

# img = cv.imread('Image_Folder_000_TEST/image8.jpg')   #Test for YOLOv5 model
# img8= cv.resize(img, (640,640), interpolation=cv.INTER_AREA)
# frame8 = detector(img8)


cv.imshow('Image0', frame0)
# cv.imshow('Image1', frame1)
# cv.imshow('Image2', frame2)
# cv.imshow('Image3', frame3)
# cv.imshow('Image4', frame4)
# cv.imshow('Image5', frame5)
# cv.imshow('Image6', frame6)
# cv.imshow('Image7', frame7)
# cv.imshow('Image8', frame8)
cv.waitKey(0)
cv.destroyAllWindows()








