# Lung-nodule-detection
We use GoogLeNet, AlexNet and ResNet to classify lung nodules. Pictures have three catagories, not nodule, bengin nodule and malignant nodule. <br>
Based on tensorflow tflearn, python3.5, GPU 1060 CUDA8.0   <br>

GoogLeNet, AlexNet and ResNet are based on the tflearn official website.<br>
all the pictures are resized to (70,70), and divided into two groups, test group and training group. They must be put in the folders like this:<br>
train---|-----0(not nodule)----------------|---picture</br>
        |                                  |---picture</br>
        |                                  |---picture</br>
        |                                   .........</br>
        |----1(benign nodule)--------------|---picture</br>
        |                                  |---picture</br>
        |                                   ..........</br>
        |----2(malignant nodule)-----------|---picture</br>
                                           |---picture</br>
                                           ...........</br>
test--- |-----0(not nodule)----------------|---picture</br>
        |                                  |---picture</br>
        |                                  |---picture</br>
        |                                   .........</br>
        |----1(benign nodule)--------------|---picture</br>
        |                                  |---picture</br>
        |                                   ..........</br>
        |----2(malignant nodule)-----------|---picture</br>
                                           |---picture</br>
                                           ...........</br>
                                           

result accuracy:<br>
GoogLeNet:94.5%     AlexNet:93%     ResNet: 91.3%<br>
 Notice:<br>
 1) If you are using windows system, need to install win_unicode_console.<br>
 2) image_preloader loading pictures, pictures are imported as list, labels are imported as one_hot vectors.
 3) Lung nodules are grayscale therefore don't have channel. Define my_func() to turn picture list into array and add channels.<br>
 4) Test part, when loading pictures you annot load 2000 pictures at once. Devide them into groups, each group have 100 or so pictures( according to the memory of the computer, the number of images per group can refer to the number of images per batch).
