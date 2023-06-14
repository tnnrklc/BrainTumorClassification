import time
import numpy as np
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename


labels = ({'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3})
labels = dict((v,k) for k,v in labels.items()) 

ws = Tk()
ws.title('Brain Tumor Classifier')
ws.geometry('1200x600') 

frame = Frame(ws, width=200, height=100)
frame.pack()
frame.place(anchor='w', relx=0.5, rely=0.5)

def uploadFiles():
    global pb1, prdct
    result.configure(text='')
    pb1 = Progressbar(
        ws, 
        orient=HORIZONTAL, 
        length=300, 
        mode='determinate'
        )
    pb1.grid(row=4, columnspan=3, pady=20)

    for i in range(5):
        ws.update_idletasks()
        pb1['value'] += 20
        time.sleep(1)

    if pb1['value'] == 100:
        prdct = Button(
        ws, 
        text='Predict', 
        command=predict
        )   
        prdct.grid(row=5, columnspan=3, pady=30)
        # Create a Label Widget to display the text or Image
        label1 = Label(frame, image = img)
        folder_path = StringVar()
        label1.grid(row=4, columnspan=5, pady=40)
        folder_path.set(file_path)
        
def open_file():
    global file_path, img
    file_path= askopenfilename(title = "Select A File", filetypes = [('jpg files', '*.jpg'), ('jpeg files', '*.jpeg'), ('png files', '*.png')])
    if file_path is not None:
        img = ImageTk.PhotoImage(Image.open(file_path))

def preprocessImage():
    global test_image1, keras
    from keras.utils import img_to_array
    from tensorflow import keras
    from tensorflow.keras.utils import load_img
    test_image1 = load_img(file_path,target_size = (224,224))
    test_image1 = img_to_array(test_image1)
    test_image1 = np.expand_dims(test_image1,axis=0)  

def predict():
    global test_image1
    from keras.models import load_model
    preprocessImage()
    cnn_model = load_model('model/brain_tumor_detection_cnn8.h5')
    result1 = np.argmax(cnn_model.predict(test_image1/255.0),axis=1)
    prob = cnn_model.predict(test_image1/255.0)
    prob = np.around(prob, decimals=2)
    predictions = [labels[k] for k in result1]
    diagnosis = f"glioma: {prob[0][0]}\nmeningioma: {prob[0][1]}\nnotumor: {prob[0][2]}\npituitary: {prob[0][3]}\nThe highest is {predictions[0]}."   
    pb1.destroy()
    result.configure(text = diagnosis)
    prdct.destroy()
    #Label(ws, text=predictions, foreground='green').grid(row=4, columnspan=3, pady=10)
    
adhar = Label(
    ws, 
    text='Upload MRI'
    )
adhar.grid(row=0, columnspan=3, padx=10)

adharbtn = Button(
    ws, 
    text ='Choose File', 
    command = lambda:open_file()
    ) 
adharbtn.grid(row=1, columnspan=3)

upld = Button(
    ws, 
    text='Upload File', 
    command=uploadFiles
    )
upld.grid(row=3, columnspan=3, pady=10)

result = Label(
    ws, 
    text='',
    foreground='green'
    )
result.grid(row=4, columnspan=5, pady=10)

ws.mainloop()
