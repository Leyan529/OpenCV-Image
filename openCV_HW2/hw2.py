# -*- coding: utf-8 -*-

import sys
from hw2_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class JPG():
	def __init__(self, source, target=None):
		self.img = cv2.imread(os.path.abspath(source))
		self.fn = os.path.basename(source)
		self.source = self.img
		self.target = cv2.imread(os.path.abspath(target))

	def orign_histogram(self):
		cv2.destroyAllWindows()
		plt.close()
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		# hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
		cv2.imshow('original image', self.img)
		plt.hist(self.gray.ravel(), 256, [0, 256], color='r')
		plt.ylabel('pixel Number')
		plt.xlabel('gray value')
		plt.title('Original image histogram')
		plt.show()

	def equalized_histogram(self):
		cv2.destroyAllWindows()
		plt.close()
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		equ = cv2.equalizeHist(self.gray)
		# hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
		cv2.imshow('Equalized image', equ)

		plt.hist(equ.ravel(), 256, [0, 256], color='r')
		plt.ylabel('pixel Number')
		plt.xlabel('gray value')
		plt.title('Equalized image histogram')
		plt.show()

	def hough_circle(self):
		cv2.destroyAllWindows()
		plt.close()
		img = self.img
		cv2.imshow(self.fn, img)

		blur = cv2.GaussianBlur(img, (5, 5), 0)
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
								   param1=5, param2=19, minRadius=14, maxRadius=20)
		circles = np.uint16(np.around(circles))
		for i in circles[0, :]:
			# draw the outer circle
			cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
			# draw the center of the circle
			cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
		cv2.imshow('Output image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def hue_histogram(self):
		try:
			cv2.destroyAllWindows()
			img = self.img
			cv2.imshow(self.fn, img)

			blur = cv2.GaussianBlur(img, (5, 5), 0)
			gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
			circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
									   param1=5, param2=19, minRadius=14, maxRadius=20)
			circles = np.uint16(np.around(circles))
			# create a mask
			mask = np.zeros(img.shape[:2], np.uint8)
			for i in circles[0, :]:
				x, y, r = i[0], i[1], i[2]
				cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)

			masked_img = cv2.bitwise_and(img, img, mask=mask)
			# cv2.imshow('44', masked_img)
			hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
			hsv = hsv[:, :, 0]
			hsv = hsv.reshape(-1)
			nonzero = []
			[nonzero.append(x) for x in hsv if x > 0]
			plt.hist(nonzero, 180, density=True)

			plt.ylabel('Probability')
			plt.xlabel('Angle')
			plt.title('Normalized Hue histogram')
			plt.show()

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def backproject(self, levels, scale):
		cv2.destroyAllWindows()
		plt.close()
		cv2.imshow(self.fn, self.source)
		hsv = cv2.cvtColor(self.source, cv2.COLOR_BGR2HSV)
		hsvt = cv2.cvtColor(self.target, cv2.COLOR_BGR2HSV)
		# calculating object histogram
		roihist = cv2.calcHist([hsv], [0, 1], None, [levels, levels], [103, 121, 48, 190])
		# normalize histogram and apply backprojection
		cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
		dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [103, 121, 48, 190], scale)
		backproj = np.uint8(dst)
		cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)
		cv2.imshow('BackProjection_result.jpg', backproj)


class Bmp():
	def __init__(self, bmp):
		self.img = cv2.imread(os.path.abspath(bmp))
		# 11*8的棋盤
		w = 11;
		h = 8
		# 設定尋找亞畫素角點的引數，採用的停止準則是最大迴圈次數30和最大誤差容限0.001
		self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
		# 獲取標定板角點的位置
		self.objp = np.zeros((w * h, 3), np.float32)
		self.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
		# 储存棋盘格角点的世界坐标和图像坐标对
		self.pattern = (w, h)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.size = self.gray.shape[::-1]
		self.fn = os.path.basename(bmp)

	def find_corner(self):
		objpoints = []  # 儲存3D點
		imgpoints = []  # 儲存2D點
		# 找到棋盘格角点
		ret, corners = cv2.findChessboardCorners(self.gray, self.pattern, flags=cv2.CALIB_CB_FAST_CHECK)
		# 如果找到足够点对，将其存储起来
		if ret == True:
			corners2 = cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
			objpoints.append(self.objp)
			imgpoints.append(corners2)
			img = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2RGB)
			# 将角点在图像上显示
			img = cv2.drawChessboardCorners(img, self.pattern, corners2, ret)
			img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2,
							 interpolation=cv2.INTER_NEAREST)
			cv2.namedWindow(self.fn, cv2.WINDOW_AUTOSIZE)
			cv2.imshow(self.fn, img)
		cv2.waitKey(500)
		cv2.destroyAllWindows()

	def intrinsic_matrix(self):
		objpoints = []  # 儲存3D點
		imgpoints = []  # 儲存2D點
		# 找到棋盘格角点
		ret, corners = cv2.findChessboardCorners(self.gray, self.pattern, None)
		# 如果找到足够点对，将其存储起来
		if ret == True:
			corners2 = cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
			objpoints.append(self.objp)
			imgpoints.append(corners2)

			# Calibration
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.size, None, None)
		# mtx # 内参数矩阵
		# dist  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
		# rvecs  # 旋转向量  # 外参数
		# tvecs  # 平移向量  # 外参数

		return mtx

	def extrinsic_matrix(self):
		objpoints = []  # 儲存3D點
		imgpoints = []  # 儲存2D點
		# 找到棋盘格角点
		ret, corners = cv2.findChessboardCorners(self.gray, self.pattern, None)
		# 如果找到足够点对，将其存储起来
		if ret == True:
			corners2 = cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
			objpoints.append(self.objp)
			imgpoints.append(corners2)
			# Calibration
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.size, None, None)
			# mtx # 内参数矩阵
			# dist  # 畸变系数   distortion cofficients = (k1,k2,p1,p2,k3)
			# rvecs  # 旋转向量  # 外参数
			# tvecs  # 平移向量  # 外参数
			rvecs = rvecs[0]
			rvecs = [rvecs[0][0], rvecs[1][0], rvecs[2][0]]
			identy = np.identity(3).tolist()


			Trans_3D = []  # 3D Translation (3 X 4)
			for i in range(0, len(identy)):
				rows = []
				for ids in identy[i]:
					rows.append(ids)
				rows.append(rvecs[i])
				Trans_3D.append(rows)
			Trans_3D = np.array(Trans_3D)

			tvecs = tvecs[0]
			tvecs = [tvecs[0][0], tvecs[1][0], tvecs[2][0]]
			Rotat_3D = []  # 3D Rotation (4 X 4)

			for i in range(0, len(tvecs)):
				rows = []
				rows.append(tvecs[i])
				rows.extend([0, 0, 0])
				Rotat_3D.append(rows)
			Rotat_3D.append([0, 1, 1, 1])
			Rotat_3D = np.array(Rotat_3D)

			extrinsic_matrix = np.dot(Trans_3D, Rotat_3D)
			return extrinsic_matrix
		# 攝像機外參矩陣：包括旋轉矩陣和平移矩陣
		# 旋轉矩陣和平移矩陣共同描述了如何把點從世界坐標系轉換到攝像機坐標系
		# 旋轉矩陣：描述了世界坐標系的坐標軸相對於攝像機坐標軸的方向
		# 平移矩陣：描述了在攝像機坐標系下，空間原點的位置

	def distortion_matrix(self):
		objpoints = []  # 儲存3D點
		imgpoints = []  # 儲存2D點
		# 找到棋盘格角点
		ret, corners = cv2.findChessboardCorners(self.gray, self.pattern, None)
		# 如果找到足够点对，将其存储起来
		if ret == True:
			corners2 = cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
			objpoints.append(self.objp)
			imgpoints.append(corners2)

			# Calibration
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.size, None, None)
			# mtx # 内参数矩阵
			# dist  # 畸变系数   distortion cofficients = (k1,k2,p1,p2,k3)
			# rvecs  # 旋转向量  # 外参数
			# tvecs  # 平移向量  # 外参数
			return np.array(dist[0])

	def draw_cube(self):
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		axis = np.float32([[2, 2, -2], [2, 0, -2], [0, 0, -2], [0, 2, -2]
							  , [2, 2, 0], [2, 0, 0], [0, 0, 0], [0, 2, 0]]).reshape(-1, 3)  # (x,y,z)軸長度
		# 找到棋盘格角点
		ret, corners = cv2.findChessboardCorners(self.gray, self.pattern, None)
		# 如果找到足够点对，将其存储起来
		if ret == True:
			corners2 = cv2.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), self.criteria)
			# Find the rotation and translation vectors.
			intrinsic_matrix = self.intrinsic_matrix()
			distortion_matrix = self.distortion_matrix()
			_, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.objp, corners2, intrinsic_matrix, distortion_matrix)
			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, intrinsic_matrix, distortion_matrix)
			print(self.fn)
			img = self.draw(self.gray, corners2, imgpts)
			img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
			cv2.namedWindow(self.fn, cv2.WINDOW_AUTOSIZE)

			dst = cv2.undistort(self.gray, intrinsic_matrix, distortion_matrix)
			dst = cv2.resize(dst, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
			# cv2.imshow('dst', dst)

			cv2.imshow(self.fn, img)

		cv2.waitKey(500)
		cv2.destroyAllWindows()

	def draw(self, img, corners, imgpts):
		red = (0, 0, 255)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		imgpts = np.int32(imgpts).reshape(-1, 2)
		img = cv2.drawContours(img, [imgpts[:4]], -1, red, thickness=20)  # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
		for i, j in zip(range(4), range(4, 8)):
			img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), red, thickness=20)
		img = cv2.drawContours(img, [imgpts[4:]], -1, red, thickness=20)
		return img


class Png():
	def __init__(self, png):
		png = os.path.abspath(png)
		self.img = cv2.imread(png)
		self.fn = os.path.basename(png)


class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.onBindingUI()

	# Write your code below
	# UI components are defined in hw1_ui.py, please take a look.
	# You can also open hw1.ui by qt-designer to check ui components.

	def onBindingUI(self):
		self.btn1_1.clicked.connect(self.on_btn1_1_click)
		self.btn1_2.clicked.connect(self.on_btn1_2_click)
		self.btn2_1.clicked.connect(self.on_btn2_1_click)
		self.btn2_2.clicked.connect(self.on_btn2_2_click)
		self.btn2_3.clicked.connect(self.on_btn2_3_click)
		self.btn3_1.clicked.connect(self.on_btn3_1_click)
		self.btn3_2.clicked.connect(self.on_btn3_2_click)
		self.btn3_3.clicked.connect(self.on_btn3_3_click)
		self.btn3_4.clicked.connect(self.on_btn3_4_click)
		self.btn4.clicked.connect(self.on_btn4_click)

	def on_btn1_1_click(self):
		jpg = os.path.abspath('images\plant.jpg')
		img = JPG(jpg)
		img.orign_histogram()

	def on_btn1_2_click(self):
		jpg = os.path.abspath('images\plant.jpg')
		img = JPG(jpg)
		img.equalized_histogram()

	def on_btn2_1_click(self):
		try:
			jpg = os.path.abspath('images\q2_train.jpg')
			img = JPG(jpg)
			img.hough_circle()

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn2_2_click(self):
		try:
			jpg = os.path.abspath('images\q2_train.jpg')
			img = JPG(jpg)
			img.hue_histogram()

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn2_3_click(self):
		try:
			group = JPG(os.path.abspath('images\q2_train.jpg'), os.path.abspath('images\q2_test.jpg'))
			group.backproject(levels=3, scale=1)
		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn3_1_click(self):
		try:
			bmps = glob('images\CameraCalibration\*.bmp')
			length = len(bmps)
			for i in range(0, length):
				bmp = os.path.abspath('images\CameraCalibration\{}.bmp'.format(i + 1))
				img = Bmp(bmp)
				img.find_corner()
			print('finished')
			cv2.destroyAllWindows()
		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn3_2_click(self):
		try:
			bmps = glob('images\CameraCalibration\*.bmp')
			length = len(bmps)
			for i in range(0, length):
				bmp = os.path.abspath('images\CameraCalibration\{}.bmp'.format(i + 1))
				fn = os.path.basename(bmp)
				img = Bmp(bmp)
				intrinsic_matrix = img.intrinsic_matrix()
				print("{} \n Intrinsic matrix: \n {}".format(fn, intrinsic_matrix))

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn3_3_click(self):
		try:
			choose = self.cboxImgNum.currentText()
			# print(choose)
			bmp = 'images\CameraCalibration\{}.bmp'.format(choose)
			fn = os.path.basename(bmp)
			img = Bmp(bmp)
			extrinsic_matrix = img.extrinsic_matrix()
			print("{} \n Extrinsic matrix: \n {}".format(fn, extrinsic_matrix))

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn3_4_click(self):
		try:
			bmps = glob('images\CameraCalibration\*.bmp')
			length = len(bmps)
			for i in range(0, length):
				bmp = os.path.abspath('images\CameraCalibration\{}.bmp'.format(i + 1))
				fn = os.path.basename(bmp)
				img = Bmp(bmp)
				distortion_matrix = img.distortion_matrix()
				print("{} \n Distortion matrix: \n {}".format(fn, distortion_matrix))

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def on_btn4_click(self):
		try:
			for i in range(0, 5):
				bmp = os.path.abspath('images\CameraCalibration\{}.bmp'.format(i + 1))
				img = Bmp(bmp)
				img.draw_cube()
			print("finished")

		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)



		### ### ###


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
