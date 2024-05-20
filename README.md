# YOLOv8 Tennis Analysis System


## Table of contents  

* [About the Program](#About-the-Program)
* [Getting Started](#Getting-Started-Steps-by-Steps)
* [How to Use the Program](##How-to-Use-the-Program)
* [Issue](#Issue)


## About the Program
一個用Python編寫的程式，旨在分析影片中的網球選手，測量其速度、球速和擊球次數。   

通過輸入的網球對打影片，該程式通過 YOLO 進行網球及球員的偵測，且利用 CNNs 提取網球場的 Key Points，接著透過 Panda 進行分析工作。  


## Getting Started - Steps by Steps
### Environments
此程式可以在 Windows 和 MacOS 系統上運行，在 Python IDE 中執行。

### Install python  
使用 Python 3 執行此程式。（[download python](https://www.python.org/downloads/)）

### Install Other Libraries  
該程式需要在 Python 中安裝
1. OpenCV 套件，通過執行以下指令來安裝：  
    
    ```shell
    pip3 install opencv-python
    ```    
2. YOLO ，通過執行以下指令來安裝： 
    ```shell
    pip3 install ultralytics
    ```  
3. RoboFlow ，通過執行以下指令來安裝： 
    ```shell
    pip3 install roboflow
    ```  
4. PyTorch
5. Numpy ，通過執行以下指令來安裝： 
    ```shell
    pip3 install numpy
    ```  
6. Pandas ，通過執行以下指令來安裝： 
    ```shell
    pip3 install pandas
    ```



## How to Use the Program

### Clone This Repo
可以使用各種源代碼管理工具，例如 Source Tree 和 GitHub Desktop 複製此存儲庫，或者執行以下指令：
```shell
git clone https://github.com/jeannie068/Tennnis_Analysis_System_YOLOv8.git
```


### Running the Program
只需執行 main.py 腳本即可產生結果，
```shell
python3 main.py
```



### Result and Test
程式執行成功後，會在相對路徑： "output_video/output_video.avi" 中看到 avi 影片，其中包含了球員速度、球員擊球球速和球員擊球次數等數據，以及球員、球的物體偵測，以及球員與球運動中的平面位置。  
  

影片最終結果如圖：
![image](output_video/screenshot.jpeg)


## Issue
- Analysis ball bouncing and show on mini-court
- Ball tracker model from YOLOv5 change to TrackNet Model

