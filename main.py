import streamlit as st
import cv2
import imutils
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'tesseract'

st.set_page_config(
    page_title="ANPR_TERA",
)

def main():
    st.title("Automatic Number Plate Recognition Web-App")
    activities = ["About", "Detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    left_column, right_column = st.columns([0.6, 0.4])

    if choice == "About":
        st.write(
    """
    **Automatic Number Plate Recognition System is license plate identification system made using OpenCV and Tesseract OCR in python. For this project, It can be used to detect number plate from image**.
    """
    )

    if choice == "Detection":
        with left_column:
            st.text(" ")
            st.markdown("Please make sure that the image is clear and clicked at the right angles so that it can be detected easily")

        uploaded_file = st.file_uploader("", type=["jpg"])
        if uploaded_file is None:
            st.text("Please upload an Image")
        else:
            image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, 1)
            image = imutils.resize(image,width=500)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray,11,17,17)
            edges = cv2.Canny(gray,170,200) #Canny Edges
            
            cnts,new=cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            img1 = image.copy()
            img1 = cv2.drawContours(img1,cnts,-1,(0,255,0),3) #4- All Contours
            
            cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]
            NumberPlateCnt=None
            img2 = image.copy()
            img2 = cv2.drawContours(img2,cnts,-1,(0,255,0),3) #5- Top 10 Contours
           
            count=0
            idx=7
            for c in cnts:
                peri=cv2.arcLength(c,True)
                approx=cv2.approxPolyDP(c,0.02*peri,True)
                if len(approx)==4:
                    NumberPlateCnt=approx
                    x,y,w,h = cv2.boundingRect(c)
                    new_img = image[y:y+h,x:x+w]
                    cv2.imwrite('Cropped image'+str(idx)+'.png',new_img)
                    idx+=1
                    break
                
            final = cv2.drawContours(image,[NumberPlateCnt],-1,(0,255,0),3)
            
            Cropped_img_loc='Cropped image7.png'
            text = pytesseract.image_to_string(Cropped_img_loc,lang='eng')

            st.image(image, caption='uploaded image')
            st.success(text)

if __name__ == '__main__':
    main()