# Note: substantial portions of this code, expecially the actual XML to CSV conversion, are credit to Dat Tran
# see his website here: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
# and his GitHub here: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# module level variables ##############################################################################################
# train and test directories
TRAINING_IMAGES_DIR = os.getcwd() + "/training_images/"
TEST_IMAGES_DIR = os.getcwd() + "/test_images/"

MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING = 10
MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING = 100

MIN_NUM_IMAGES_REQUIRED_FOR_TESTING = 3

# output .csv file names/locations
TRAINING_DATA_DIR = os.getcwd() + "/" + "training_data"
TRAIN_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "train_labels.csv"
EVAL_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "eval_labels.csv"

#######################################################################################################################
def main():
    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    # if the training data directory does not exist, create it
    try:
        if not os.path.exists(TRAINING_DATA_DIR):
            os.makedirs(TRAINING_DATA_DIR)
        # end if
    except Exception as e:
        print("unable to create directory " + TRAINING_DATA_DIR + "error: " + str(e))
    # end try


    # convert training xml data to a single .csv file
    print("converting xml training data . . .")
    trainCsvResults = xml_to_csv(TRAINING_IMAGES_DIR)
    trainCsvResults.to_csv(TRAIN_CSV_FILE_LOC, index=None)
    print("training xml to .csv conversion successful, saved result to " + TRAIN_CSV_FILE_LOC)

    # convert test xml data to a single .csv file
    print("converting xml test data . . .")
    testCsvResults = xml_to_csv(TEST_IMAGES_DIR)
    testCsvResults.to_csv(EVAL_CSV_FILE_LOC, index=None)
    print("test xml to .csv conversion successful, saved result to " + EVAL_CSV_FILE_LOC)

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TRAINING_IMAGES_DIR):
        print('')
        print('ERROR: the training images directory "' + TRAINING_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the training images?')
        print('')
        return False
    # end if

    # get a list of all the .jpg / .xml file pairs in the training images directory
    trainingImagesWithAMatchingXmlFile = []
    for fileName in os.listdir(TRAINING_IMAGES_DIR):
        if fileName.endswith(".jpg"):
            xmlFileName = os.path.splitext(fileName)[0] + ".xml"
            if os.path.exists(os.path.join(TRAINING_IMAGES_DIR, xmlFileName)):
                trainingImagesWithAMatchingXmlFile.append(fileName)
            # end if
        # end if
    # end for

    # show an error and return false if there are no images in the training directory
    if len(trainingImagesWithAMatchingXmlFile) <= 0:
        print("ERROR: there don't seem to be any images and matching XML files in " + TRAINING_IMAGES_DIR)
        print("Did you set up the training images?")
        return False
    # end if

    # show an error and return false if there are not at least 10 images and 10 matching XML files in TRAINING_IMAGES_DIR
    if len(trainingImagesWithAMatchingXmlFile) < MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING:
        print("ERROR: there are not at least " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING) + " images and matching XML files in " + TRAINING_IMAGES_DIR)
        print("Did you set up the training images?")
        return False
    # end if

    # show a warning if there are not at least 100 images and 100 matching XML files in TEST_IMAGES_DIR
    if len(trainingImagesWithAMatchingXmlFile) < MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING:
        print("WARNING: there are not at least " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " images and matching XML files in " + TRAINING_IMAGES_DIR)
        print("At least " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " image / xml pairs are recommended for bare minimum acceptable results")
        # note we do not return false here b/c this is a warning, not an error
    # end if

    if not os.path.exists(TEST_IMAGES_DIR):
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        return False
    # end if

    # get a list of all the .jpg / .xml file pairs in the test images directory
    testImagesWithAMatchingXmlFile = []
    for fileName in os.listdir(TEST_IMAGES_DIR):
        if fileName.endswith(".jpg"):
            xmlFileName = os.path.splitext(fileName)[0] + ".xml"
            if os.path.exists(os.path.join(TEST_IMAGES_DIR, xmlFileName)):
                testImagesWithAMatchingXmlFile.append(fileName)
            # end if
        # end if
    # end for

    # show an error and return false if there are not at least 3 images and 3 matching XML files in TEST_IMAGES_DIR
    if len(testImagesWithAMatchingXmlFile) <= 3:
        print("ERROR: there are not at least " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TESTING) + " image / xml pairs in " + TEST_IMAGES_DIR)
        print("Did you separate out the test image / xml pairs from the training image / xml pairs?")
        return False
    # end if

    return True
# end function

#######################################################################################################################
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                     int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
