
import numpy as np 
import cv2

# Taken from Adrian Rosebrock and adapted for our needs

def gamma_correct(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    # The pixel intensities are scaled from the range [0, 255] to [0, 1.0]
    # Perform gamma correction
    # Scaling back the image to range [0, 255]
    gamma_inv = 1.0 / gamma
    lookup_table = np.array([((it / 255.0) ** gamma_inv) * 255
        for it in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, lookup_table)

def equalize_light(image, limit=2, grid=(8,8), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image) 

def main():
    cap = cv2.VideoCapture('Input_4.mp4')
    #gamma = 1.5#Change Gamma value and check
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Output_4.mp4', fourcc, 30, (1920, 1080))
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


    while(cap.isOpened()):
        ret, original = cap.read()
        if ret==False:
            break
        corrected=equalize_light(original)
        #corrected = gamma_correct(corrected, gamma)
        corrected = cv2.GaussianBlur(corrected,(5,5),0)
        corrected=cv2.medianBlur(corrected,5)
        out.write(np.hstack([corrected]))
        cv2.namedWindow('Video improvement',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video improvement', 1920,1080)
        cv2.imshow('Video improvement', np.hstack([corrected]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        
main()
