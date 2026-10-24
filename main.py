import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showerror

from PIL import Image , ImageTk
import cv2

from tkinter.ttk import Treeview, Scrollbar

from ultralytics import YOLO
import time
import os
import numpy as np

class Application(Tk):
    # 构造函数
    def __init__(self):
        Tk.__init__(self)
        self.title('基于深度学习的水稻病害智能识别系统')
        self.geometry('800x600+300+100')
        # 窗口限定 false: 宽度不变 false: 高度不变
        self.resizable(False , False)
        self.config(bg='#87CEEB')


        # 存储所有图片的标注对象
        self.result_list = []
        # 初始化ui界面
        self.initUI()
        # 加载模型
        self.init_model()

    # 初始化UI界面
    def initUI(self):
        # 标题文本
        self.lblTitle = Label(self , text='基于深度学习的水稻病害智能识别系统' , font=("宋体" , 20) , bg='#87CEEB')
        self.lblTitle.pack(pady=20)
        # 水平线
        self.line = Frame(self , width=780 , height= 1 , bg='#B1B1B1')
        self.line.pack()
        # 中间区域内容
        self.mainFrame = Frame(self , bg='#87CEEB')
        self.mainFrame.pack(pady=10)
        # 显示图片
        imageW = 310
        imageH = 310
        self.photo = self.get_image_icon('res/12.png' , imageW , imageH)
        self.lblImage = Label(self.mainFrame , width=imageW + 170 , height=imageH , image=self.photo , bg='#FFFFFF')
        self.lblImage.pack(side=LEFT , padx=5)

        # 中间区域右侧显示内容
        self.mainRightFrame = Frame(self.mainFrame , width=280 , height=200 , bg="#87CEEB")
        self.mainRightFrame.pack(side=LEFT)

        # 识别结果显示区域
        self.resultFrame = LabelFrame(self.mainRightFrame , width=280 , height=160 , text='识别结果' , bg='#87CEEB')
        self.resultFrame.pack(padx=5)
        self.resultFrame.grid_propagate(False)
        # 识别结果显示区域 - 识别结果
        self.lblResult1 = Label(self.resultFrame , text='暂无结果' , font=('宋体' , 30) , foreground='#E20D00' , bg='#87CEEB')
        self.lblResult1.grid(row=0 , column=0 , columnspan=4 , pady = 20)

        # 识别结果显示区域 - 置信度文本
        self.lblResult2 = Label(self.resultFrame , width=10 , anchor='e' , text='置信度' , bg='#87CEEB')
        self.lblResult2.grid(row = 1 , column=0)

        # 识别结果显示区域 - 置信结果
        self.lblResult = Label(self.resultFrame, width=10, anchor='w', text='', bg='#87CEEB' , foreground = "#E20D00")
        self.lblResult.grid(row=1, column=1)

        # 识别结果显示区域 - 用时文本
        self.lblResult3 = Label(self.resultFrame, text='用时', bg='#87CEEB')
        self.lblResult3.grid(row=1, column=2)

        # 识别结果显示区域 - 用时结果
        self.lblResult4 = Label(self.resultFrame, width=10, anchor='w',text='', bg='#87CEEB' , foreground = '#E20D00')
        self.lblResult4.grid(row=1, column=3)

        # 操作显示区域
        self.optFrame = LabelFrame(self.mainRightFrame , width = 280 , height = 160 , text = '操作' , bg = '#87CEEB')
        self.optFrame.pack(padx = 5)
        self.optFrame.grid_propagate(False)
        # 操作显示区域-打开图片
        self.openIcon = self.get_image_icon('res/ui_images/img.png')
        self.openImageBtn = Button(self.optFrame , width = 90 , height = 25 ,
                                   image = self.openIcon , compound = LEFT , text = '打开图片' , command = self.open_image)
        self.openImageBtn.grid(row = 0 , column = 0 , padx = 20 , pady = 20)

        # 操作显示区域-打开文件夹
        self.openDirIcon = self.get_image_icon('res/ui_images/folder.png')
        self.openDirBtn = Button(self.optFrame, width=90, height=25,
                                   image=self.openDirIcon, compound=LEFT, text='打开文件夹', command=self.open_dir)
        self.openDirBtn.grid(row=0, column=1)

        # 操作显示区域-打开视频
        self.openVideoIcon = self.get_image_icon('res/ui_images/video.png')
        self.openVideoBtn = Button(self.optFrame,width=90, height=25,
                                   image=self.openVideoIcon, compound=LEFT, text='打开视频', command=self.open_video)
        self.openVideoBtn.grid(row=1, column=0)

        # 操作显示区域-打开摄像头
        self.openCameraIcon = self.get_image_icon('res/ui_images/camera.png')
        self.openCamBtn = Button(self.optFrame, width=90, height=25,
                                 image=self.openCameraIcon, compound=LEFT, text='打开摄像头', command=self.open_camera)
        self.openCamBtn.grid(row=1, column=1)

        # 表格区域
        self.tableFrame = LabelFrame(self , width = 770 , height = 160 , text = '识别结果信息' , bg = '#87CEEB')
        self.tableFrame.pack()
        self.tableFrame.pack_propagate(False)

        from tkinter import ttk

        # 2. 创建树状列表（表格）控件，父容器为tableFrame
        self.result_table = ttk.Treeview(self.tableFrame, columns=('id', 'path', 'result', 'trust'), show='headings')

        # 3. 设置表头（列名）
        self.result_table.heading('id', text='序号')
        self.result_table.heading('path', text='路径')
        self.result_table.heading('result', text='结果')
        self.result_table.heading('trust', text='置信度')

        # 4. 设置每列宽度
        self.result_table.column('id', width=50, anchor='center')  # anchor可选，设置内容居中
        self.result_table.column('path', width=395, anchor='w')  # 左对齐，适合长文本
        self.result_table.column('result', width=150, anchor='center')
        self.result_table.column('trust', width=150, anchor='center')

        # 5. 用pack方式摆放表格，填充整个容器
        self.result_table.pack(fill='both', expand=True)

        # self.result_table.insert("", END,
        #                          values=(1, "E:/datasets/Data/train/Blast/BLAST1_003.jpg", "褐斑病", "99.99%"))
        # self.result_table.insert("", END,
        #                          values=(2, "E:/datasets/Data/train/Blast/BLAST1_004.jpg", "白叶枯病", "69.12%"))
        # self.result_table.insert("", END,
        #                          values=(3, "E:/datasets/Data/train/Blast/BLAST1_005.jpg", "褐斑病", "99.99%"))
        # self.result_table.insert("", END,
        #                          values=(4, "E:/datasets/Data/train/Blast/BLAST1_006.jpg", "稻瘟病", "89.67%"))
        # self.result_table.insert("", END,
        #                          values=(5, "E:/datasets/Data/train/Blast/BLAST1_007.jpg", "枯草病", "72.37%"))
        # self.result_table.insert("", END,
        #                          values=(6, "E:/datasets/Data/train/Blast/BLAST1_008.jpg", "褐斑病", "39.93%"))

        self.rightScrollBar = Scrollbar(self.tableFrame, orient="vertical", command=self.result_table.yview)
        self.rightScrollBar.pack(fill=Y, side=RIGHT)
        self.result_table.config(yscrollcommand=self.rightScrollBar.set)

        self.result_table.bind('<ButtonRelease-1>' , self.tree_click)

    # 加载模型
    def init_model(self):
        # 加载检测模型
        self.model = YOLO("models/best.pt" , task='classify')

    # 获得图片图标，并更改大小
    def get_image_icon(self , path , width=20 , height = 20):
        image = Image.open(path)
        # 获取图片原始的宽度和高度
        oldWidth = image.width
        oldHeight = image.height
        # 变换宽度和高度
        newWidth = int(oldWidth * (height / oldHeight))
        newHeight = int(height)

        # 更改图片大小
        image = image.resize((newWidth , newHeight))
        # 转换为tk可以使用的格式
        phone = ImageTk.PhotoImage(image)
        return phone

    #检索图片中的水稻
    def check_image(self , img , filePath):
        # 记录起始时间
        t1 = time.time()

        #检测
        self.result = self.model(img, conf=0.3)[0]
        # 包含四个小数，分别代表【‘百叶枯病’ ， ‘稻瘟病’ ， ‘褐斑病’ ， ‘枯草病’】四种情况
        self.conf_list = self.result.probs.data.tolist()

        cls_index = self.result.probs.top1
        # 获得最大准确度【置信度】
        conf = self.conf_list[cls_index]

        if conf > 0.3:
            # 显示水稻病
            show_conf = '{:.2f}%'.format(conf * 100)
            CH_names = ['白叶枯病', '稻瘟病', '褐斑病', '枯草病']
            res_name = CH_names[cls_index]

        else:
            show_conf = '100%'
            res_name = '无法检测'

        # 记录结束时间
        t2 = time.time()

        # 显示用时
        take_time = "{:.3f} s".format(t2 - t1)
        # 添加表格信息
        # 序号（当前记录数 + 1）
        num = len(self.result_table.get_children()) + 1
        self.result_table.insert("", tkinter.END, values=(num, filePath, res_name, show_conf))
        # 存储到列表中
        dict = {
            "take_time": take_time,
            "show_conf": show_conf,
            "res_name": res_name,
            "filePath": filePath,
            "img": img
        }
        self.result_list.append(dict)
        # 显示窗口信息
        self.show_window_info(dict)


    def show_window_info(self , dict):
        # 显示时间
        self.lblResult4.config(text = dict["take_time"])
        #显示置信度
        self.lblResult2.config(text=dict["show_conf"])
        # 显示水稻病名称
        self.lblResult1.config(text=dict["res_name"])
        # 转换色彩空间，从BGR转到RGB
        now_img = cv2.cvtColor(dict['img'] , cv2.COLOR_BGR2RGB)
        # 更改图片的大小
        image = Image.fromarray(now_img)
        oldWidth = image.width
        oldHeight = image.height
        newWidth = int(oldWidth * (310 / oldHeight))
        newHight = int(310)
        self.image = image.resize((newWidth , newHight))
        self.photo = ImageTk.PhotoImage(self.image)
        self.lblImage.config(image = self.photo)

    # 打开图片按钮
    def open_image(self):
        # 文件选择对话框
        filePath = filedialog.askopenfilename(
            filetypes=[('打开文件' , '*.png') , ('打开文件' , '*.jpg'), ('打开文件' , '*.gif'), ('打开文件' , '*.jpeg')]
        )
        if filePath:
            # 清空表格
            for item in self.result_table.get_children():
                self.result_table.delete(item)
            # 清空列表
            self.result_list.clear()
            # 读取图片
            img = cv2.imdecode(np.fromfile(filePath , dtype=np.uint8) , cv2.IMREAD_COLOR)
            # 检测
            self.check_image(img , filePath)

    # 打开文件夹
    def open_dir(self):
        # 文件夹对话框
        dirPath = filedialog.askdirectory()
        if dirPath:
            # 文件名列表
            fileList = os.listdir(dirPath)
            # 清空表格
            for item in self.result_table.get_children():
                self.result_table.delete(item)
            # 清空列表
            self.result_list.clear()
            #遍历所有图片
            for v in fileList:
                # 获取文件拓展名
                ext = v.split('.')[-1]
                # 判断文件类型
                if ext == 'jpg' or ext == 'png' or ext == 'JPG' or ext == 'PNG':
                    # 图片的完整地址
                    filePath = dirPath + '/' + v
                    # 判断是否为文件
                    if os.path.isfile(filePath):
                        # 读取图片
                        img = cv2.imdecode(np.fromfile(filePath , dtype=np.uint8) , cv2.IMREAD_COLOR)
                        # 检测
                        self.check_image(img , filePath)


    # 打开视频按钮
    def open_video(self):
        # 文件选择对话框
        filePath = filedialog.askopenfilename(filetypes=[('打开文件' , '*.mp4')])
        if filePath:
            # 清空表格
            for item in self.result_table.get_children():
                self.result_table.delete(item)
            # 清空列表
            self.result_list.clear()
            # 捕获视频
            cap = cv2.VideoCapture(filePath)
            # 获取视频总帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            # 读取视频
            while True:
                # 计算当前读取的帧
                current_frame_number = idx * 100
                if current_frame_number > frame_count:
                    current_frame_number = frame_count - 1
                # 设置将要读取的帧
                cap.set(cv2.CAP_PROP_POS_FRAMES , current_frame_number)
                idx += 1
                # 读取一帧
                res , frame = cap.read()
                # 设置当前图像对应的路径
                new_filePath = filePath + '[' + str(idx) + ']'
                # 检测
                self.check_image(frame , new_filePath)
                # 结束退出
                if res == False or current_frame_number == frame_count - 1:
                    break

    # 打开摄像头
    def open_camera(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        # 读取一帧
        ret  , frame = self.cap.read()
        # 释放摄像头
        self.cap.release()
        # 设置文件路劲显示内容
        filePath = '摄像头'
        # 清空表格
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        # 清空对象列表
        self.result_list.clear()
        # 检测图片
        self.check_image(frame , filePath)

    # 鼠标点击表格
    def tree_click(self , event):
        if len(self.result_table.selection()) > 0:
            # 获得选中项
            item = self.result_table.selection()[0]
            # 获取该项的文本
            values = self.result_table.item(item, 'values')
            # 重新设置窗口信息
            dict = self.result_list[int(values[0])-1]
            self.show_window_info(dict)


app = Application()
app.mainloop()
