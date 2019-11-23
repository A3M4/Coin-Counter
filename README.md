# Coin-Counter-based-on-TensorFlow-Object-Detection-API-Tutorial
TensorFlow’s Object Detection API gives access to anyone who wants to quickly create a image/video recognition software. I will be building a coin counter to detect if there is a coin in the image and calculate the total amount of money simultaneously. The following pictures show the prototype's results, and future improvements of this project will be discussed in the end of this article.

<img src="https://i.ibb.co/wwgYtX7/35028879540694746.jpg" alt="avatar" style="zoom: 200%;" />

<img src="https://i.ibb.co/SssJyFt/3768238625917334.jpg" alt="avatar" style="zoom: 200%;" />

# Overview

TensorFlow-GPU is quite faster than CPU for the training process, but it only supports NVIDIA GPU, this criterion is not met on my laptop. Therefore I chose TensorFlow-CPU and a lot of time was sacrificed to meet a good result. The newest TensorFlow  CPU version(2.0) has some incompatibilities with the object detection models, after trying different versions, TensorFlow 1.15.0 is the most appropriate version to use with my environment(Python3.7.0, Win10-x64). The diagram below shows the workflow for this project.

<img src="https://i.ibb.co/7nCXpvX/objectdetection-1.png" alt="avatar" style="zoom: 200%;" />

<br/>

# Configuring the Environment

Tensorflow can be installed by typing the following code in Command Prompt window under C:\ directory:

```pip3
pip3 install --upgrade tensorflow==1.15.0
```

TensorFlow Object Detection API readme installation instructions can be found on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md, follow this readme to install required dependencies. One thing to note is that it doesn't show protobuf-compiler installation for windows users, you can find the zip file on https://github.com/protocolbuffers/protobuf/releases. Then clone the TensorFlow models repository(https://github.com/tensorflow/models) and save it under the same directory with TensorFlow.

In the directory"models\research\object_detection\protos", there are a lot files ending in .proto. These files can be transferred into *.py by protobuf-compiler, navigate into models\research and type the following command line(protoc directory must be set in PATH in system variables):

```
protoc-3.10.1-win64\bin\protoc.exe object_detection/protos/*.proto --python_out=.
```

After finishing compiling, there will be a corresponding .py file for every .proto file under the protos directory. Before going through next step, add a variable in system variables with the name PYTHONPATH and then add those values below to PYTHONPATH:

```
C:\TensorFlow\models

C:\TensorFlow\models\research

C:\TensorFlow\models\research\slim

C:\TensorFlow\models\research\object_detection

C:\Users\Python\python37
```

Then add %PYTHONPATH% into PATH variable. Now the environment is prepared appropriately. Use Jupyter Notebook to open the file named object_detection_tutorial.ipynb in "models\research\object_detection", click on the Jupyter toolbar and select the Run All, you should see two sample images when the program stops.

<br/>

# Labelling and Creating Dataset

In order not to confuse the model and save time, the tail side of Canadian $1 and $2 coin was used for this project, and a total of 428 pictures in various backgrounds were taken and compressed into smaller size around 300kb. 

**LabelImg** is a easy-to-use graphical image annotation tool, it can be downloaded on https://tzutalin.github.io/labelImg/. The user interface is shown below, just open a directory of saved images, then use Create RecBox at the left side to select coins and label it then click save.

<img src="https://i.ibb.co/qCqZQ1c/2019-11-21-111721.jpg" alt="avatar" style="zoom: 200%;" />

when finishes and a XML file containing coordinate data will be generated for each corresponding JPG file.  the part of XML file should look like this:

```
-<object>
<name>Toonie</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>

-<bndbox>
<xmin>205</xmin>
<ymin>156</ymin>
<xmax>344</xmax>
<ymax>299</ymax>

</bndbox>
</object>

-<object>
<name>Toonie</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>

-<bndbox>
<xmin>225</xmin>
<ymin>369</ymin>
<xmax>365</xmax>
<ymax>509</ymax>

</bndbox>
</object>
</annotation>
```

From the above code, it is clear that two Toonies are labeled in the bndbox, and the four three-digit numbers represent the coordinates of four corners of the bndbox. After the labelling, create a "training_images" folder and a "test_images" folder, then copy all the pictures and XML files into "training_images" and randomly select and move 10 pictures of loonie and 10 pictures of toonie into "test_images".

<img src="https://i.ibb.co/rsWVbr2/2019-11-22-223405.jpg" alt="avatar" style="zoom: 200%;" />

The next step(use xml_to_csv.py) is generating two CSV files(one for training and one for testing) from XML files, of which the part of data table as pictured above. This is primarily performed by **xml.etree.ElementTree**, which is a powerful module implementing a efficient API for parsing and creating XML data. The following function uses **glob.glob** to return all paths of XML files and  **xml.etree.ElementTree** to parse them, then save the data into pandas' dataframe with eight columns, each row represents a coin's position.

```python
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                     int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename','width','height','class','xmin','ymin','xmax','ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
```

After that, use generate_tfrecords.py to create two tfrecord files. Run the script and check "C:\TensorFlow_Tut_3_Object_Detection_Walk-through-master\training_data", there should be two files called eval.tfrecord and train.tfrecord. Besides, label_map.pbtxt should be edited based on the object you want to train. For the purposes of this article, the label_map.pbtxt file looks like this:

```
item {
  id: 1
  name: 'Toonie'
  display_name: 'Toonie'
}
item {
  id: 2
  name: 'Loonie'
  display_name: 'Loonie'
}
```

The final part of this section is to choose a model from the Tensorflow detection model zoo(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), more than 20 pre-trained models are listed on this link. The **ssd_inception_v2_coco** is usually a good choice for images with different objects, which is the one used in this project(as for dynamic object like webcam or video, **ssd_mobilenet_v1_coco** is the right one to use). Download this model to the repository directory and extract it. 

**ssd_inception_v2_coco.config** can be found on https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config. Save it into the repository directory and  edit the path in train_input_reader and eval_input_reader to the corresponding path of tfrecord.  So far, the preparing process is completed and ready to train.

<br/>

# Training the Model

**Batch_size** in the config file means the number of images you feed to train at a time, which influences the training time and gradient shifts. It might return error like "*Allocation of X exceeds 10% of system memory*" if the number is too large. For the best performance batch size, there is no general principle to follow, you can try different batch sizes within short steps and pick one works best.  

The number of steps is defined as: **num_steps**=(epoch*examples)/batch_size. Therefore, epoch is a definite number defined by the other three variables and one epoch represents a complete round of using all training data for calculation and optimizations.

In consideration of computing power of ordinary laptops and  time costs, **batch_size: 20** and **num_steps: 18000** is set in **ssd_inception_v2_coco.config**. In this case, it usually takes around 40 hours for ordinary PCs to finish the job. After running train.py, the dataset returned by the script is plotted onto a scatter diagram. The plot below has 18000 dots, each dot shows the value of **Loss** on certain **Step**, it has fast convergence before 2500 steps. After that, the speed of decreasing of loss becomes slow, but the model is still becoming more accurate. Generally, the lower the Loss, the higher the model's accuracy, for this training, the final value of loss remains stable around 0.6 to 1.0.  <img src="https://i.ibb.co/HGpFcwW/out.png" alt="avatar" style="zoom: 200%;" />

<br/>

# Testing the Model

Model.ckpt will be generated when train.py finishes running, this file is used for producing inference graph by running export_inference_graph.py. In the test.py file, the block of code below is added to put text on graphs:

```python
coin = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0,         index] > 0.75]
ToonieCount = []
LoonieCount = []
for i in range(0, int(len(coin))):
    if str(coin[i]['name']) == "Toonie":
        ToonieCount.append("t")
    else:
        LoonieCount.append("l")
        
# Singular & Plural
toonie = " Toonies"
loonie = " Loonies"
if len(ToonieCount) <= 1:
    toonie = " Toonie"
if len(LoonieCount) <= 1:
    loonie = " Loonie"

# Add text to image               
text = "You have " + str(len(ToonieCount)) + toonie + " and " + str(len(LoonieCount)) +           loonie + ", " + str(len(ToonieCount)+len(ToonieCount)+len(LoonieCount)) + " Bucks         in Total."
                    
cv2.putText(img=Image_Coin_Counter, text=text, org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.1, 
            color=[0, 0, 0], lineType=4, thickness=8)

cv2.putText(img=Image_Coin_Counter, text=text, org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.1, 
            color=[255, 255, 255], lineType=4, thickness=2)
                            
cv2.imwrite(TEST_IMAGE_DIR + r'/' + str(random.random())[2:] + r'.jpg', 			                 Image_Coin_Counter)                    
```

On the first line, coin defines a list of dictionaries, each dictionary represents an object that has been detected, for an example:

```python
[{'id': 1, 'name': 'Toonie'}, {'id': 2, 'name': 'Loonie'}]
```

This list is used to count the number of coins and decide whether nouns are plural or singular. 

Create a folder called "final_test_images" and move the original photos you want to test into it before running test.py. Up to now, the repository directory we are working with looks like the picture below:

<img src="https://i.ibb.co/HH7sZrr/2019-11-22-234629.jpg" alt="avatar" style="zoom: 100%;" />

Through testing, it can be found that the model is smart enough to detect the head side of both coins even if it was trained only using the tail side of coins:

<img src="https://i.ibb.co/ZdWCXRb/67876047806213.jpg" alt="avatar" style="zoom: 200%;" />

<br/>

# Future Improvements 

Firstly, in order to put the model into practical application, different variety of coins must be included in training, including commemorative coins, paper currency and foreign currency. API or web crawler can be used for calculating real time exchange rates.

Secondly, a much larger range of data is desirable for accuracy in creating a coin counter, 428 pictures are far from perfect. And, more pictures of overlapped coins need to be collected for a better accuracy in the case of various coins overlap with each other. 

At last, it is obviously that more training steps will give a better result, it takes time on personal computer   so the training part can be running on a remote server with strong computing capability.

