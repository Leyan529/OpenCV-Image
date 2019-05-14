# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import os
import numpy as np
import copy
import math


class Img():
	def __init__(self, bmp):
		self.img = cv2.imread(os.path.abspath(bmp))
		self.fn = os.path.basename(bmp)

	def load(self):
		cv2.destroyAllWindows()
		rows, cols, _ = self.img.shape
		print('Height = {}'.format(rows))
		print('Width = {}'.format(cols))
		cv2.imshow(self.fn, self.img)

	def color_conver(self):
		cv2.destroyAllWindows()
		cv2.imshow(self.fn, self.img)
		img = self.img
		b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
		g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
		r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

		b[:, :] = img[:, :, 0]
		g[:, :] = img[:, :, 1]
		r[:, :] = img[:, :, 2]
		cv2.imshow('Mixed Color', cv2.merge([g, r, b]))

	def Flipping(self):
		cv2.destroyAllWindows()
		cv2.imshow(self.fn, self.img)
		flipImg = cv2.flip(self.img, 1)
		cv2.imshow('Flip', flipImg)
		cv2.imwrite('images/dog_flip.bmp', copy.deepcopy(flipImg))

	def blending(self):
		def blend_img(x):
			a_img = copy.deepcopy(self.img)
			b_img = cv2.imread('images/dog_flip.bmp')
			weight = float(x / 100)
			blend = cv2.addWeighted(a_img, 1 - weight, b_img, weight, 0)
			cv2.imshow(window_name, blend)

		window_name = 'BLENDING'
		cv2.destroyAllWindows()
		cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
		# create trackbars for color change
		cv2.createTrackbar('BLEND', window_name, 0, 100, blend_img)
		cv2.imshow(window_name, self.img)

	def edge_detect(self):
		try:
			def Sobel_calc(A):
				try:
					Wx = np.outer([1, 2, 1], [-1, 0, 1])  # Vertical Sobel detect
					Wy = np.transpose(Wx)  # Horizontal Sobel detect
					print(Wx)
					rows, cols, _ = self.img.shape
					Gx = np.zeros((rows, cols))  # Vertical edge detect
					Gy = np.zeros((rows, cols))  # Horizontal edge detect
					G = np.zeros((rows, cols))

					for i in range(1, rows - 2):
						for j in range(1, cols - 2):
							mat = A[i - 1:i + 2, j - 1:j + 2]
							Gx[i][j] = np.dot(np.reshape(Wx, -1), np.reshape(mat, -1))
							Gy[i][j] = np.dot(np.reshape(Wy, -1), np.reshape(mat, -1))
							G[i][j] = math.sqrt(Gx[i][j] ** 2 + Gy[i][j] ** 2)

					Gx = np.uint8(np.absolute(Gx))  # Gradients_X
					Gy = np.uint8(np.absolute(Gy))  # Gradients_Y
					G = np.uint8(np.absolute(G))  # Gradients

					return Gx, Gy, G
				# print(type(self.img))
				except Exception as e:
					traceback = sys.exc_info()[2]
					# logger.error(sys.exc_info())
					print(traceback.tb_lineno)
					print(e)

			cv2.destroyAllWindows()
			gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
			guass_blur = cv2.GaussianBlur(gray, (3, 3), 0)  # 3 X 3 GaussianBlur matrix

			sobel_vert, sobel_horz, sobel = Sobel_calc(copy.deepcopy(guass_blur))
			cv2.imshow(self.fn, gray)
			cv2.imshow('vertical edges', sobel_vert)
			cv2.imshow('Horizontal edges', sobel_horz)

			def calc_man(x):
				img = sobel
				img = np.uint8(np.absolute(img))
				ret, img = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
				cv2.imshow('Magnitude', img)

			# create trackbars for Magnitude
			Windowname = 'Magnitude'
			cv2.namedWindow(Windowname)
			cv2.createTrackbar('Magnitude', Windowname, 40, 255, calc_man)  # Initial Magnitude for 40
			cv2.imshow(Windowname, sobel)

			gx = np.float32(copy.deepcopy(sobel_vert))
			gy = np.float32(copy.deepcopy(sobel_horz))
			phase = cv2.phase(gy, gx, angleInDegrees=True)
			print(phase)

			def calc_ang(x):
				gx = np.float32(sobel_vert)
				gy = np.float32(sobel_horz)
				phase = cv2.phase(gy, gx, angleInDegrees=True)
				rows, cols = phase.shape
				img = copy.deepcopy(sobel)
				for i in range(rows):
					for j in range(cols):
						if not (phase[i][j] > (x - 10) and phase[i][j] < (x + 10)):
							img[i][j] = 0
				cv2.imshow('Direction', img)

			# create trackbars for Magnitude
			Windowname = 'Direction'
			cv2.namedWindow(Windowname)
			cv2.createTrackbar('Angle', Windowname, 0, 360, calc_ang)
			cv2.imshow(Windowname, sobel)


		except Exception as e:
			traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			print(traceback.tb_lineno)
			print(e)

	def Gauss_and_Laplac(self):

		cv2.destroyAllWindows()

		gp_L0 = self.img
		gp_L1 = cv2.pyrDown(cv2.GaussianBlur(gp_L0, (5, 5), 0))  # Gaussian(3x3) pyramid level 1 image
		cv2.imshow('Gaussian(3x3) pyramid level 1 ', gp_L0)

		gp_L1 = cv2.GaussianBlur(gp_L1, (5, 5), 0)  # 3 X 3 GaussianBlur matrix
		gp_L2 = cv2.pyrDown(gp_L1)

		UP_gp_L2 = cv2.GaussianBlur(cv2.pyrUp(gp_L2), (5, 5), 0)
		Lp_L1 = gp_L1 - UP_gp_L2
		cv2.imshow('Laplacian pyramid level 1 image', Lp_L1)

		UP_gp_L1 = cv2.GaussianBlur(cv2.pyrUp(gp_L1), (5, 5), 0)
		Lp_L0 = gp_L0 - UP_gp_L1
		cv2.imshow('Laplacian pyramid level 0 image', Lp_L0)

		Igp_L2 = gp_L2
		UP_Igp_L2 = cv2.GaussianBlur(cv2.pyrUp(Igp_L2), (5, 5), 0)
		Igp_L1 = Lp_L1 + UP_Igp_L2  # Inverse Gaussian pyramid level 1
		cv2.imshow('Inverse Gaussian pyramid level 1', Igp_L1)

		Up_Igp_L1 = cv2.GaussianBlur(cv2.pyrUp(Igp_L1), (5, 5), 0)
		Igp_L0 = Up_Igp_L1 + Lp_L0
		cv2.imshow('Inverse Gaussian pyramid level 0', Igp_L0)

	def glo_hold(self):
		cv2.destroyAllWindows()
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		cv2.imshow(self.fn, self.img)
		ret, global_thres = cv2.threshold(self.img, 80, 255, cv2.THRESH_BINARY)
		cv2.imshow('Thresholded image', global_thres)

	def loc_hold(self):
		cv2.destroyAllWindows()
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		cv2.imshow(self.fn, self.img)

		local_thres = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 0)
		cv2.imshow('Adaptive Thresholded image', local_thres)

	def show_Transform1(self, angle, scale, tx, ty):
		try:
			cv2.destroyAllWindows()
			rows, cols, _ = self.img.shape
			# a = scale * math.cos(angle)
			# b = scale * math.sin(angle)
			# x_new = 130 + tx
			# y_new = 125 + ty
			# trans_matrix = []
			# trans_matrix.append([a, b, ((1 - a) * x_new - b * y_new)])
			# trans_matrix.append([-b, a, (b * x_new + (1 - a) * y_new)])
			# trans_matrix = np.array(trans_matrix)
			# print(trans_matrix)
			trans_matrix = cv2.getRotationMatrix2D(center=(130 + tx, 125 + ty), angle=angle, scale=scale)
			img1 = cv2.warpAffine(self.img, trans_matrix, (cols, rows))
			cv2.namedWindow(self.fn, cv2.WINDOW_AUTOSIZE)
			cv2.imshow(self.fn, img1)

		except Exception as e:
			traceback = sys.exc_info()[2]
			print(traceback.tb_lineno)
			print(e)

	def show_Transform2(self):
		cv2.destroyAllWindows()
		rect_pts = []  # Starting and ending points
		hint = ['左上', '右上', '右下', '左下']
		window_name = 'Original Perspective'
		print('左上')
		self.img = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

		def select_points(event, x, y, flags, param):
			def mappImg(points, img):

				pst1 = np.float32(points)
				pst2 = np.float32([[20, 20], [450, 20], [450, 450], [20, 450]])

				M = cv2.getPerspectiveTransform(pst1, pst2)
				dst = cv2.warpPerspective(img, M, (430, 430))
				cv2.imshow('Perspective Result Image', dst)
				cv2.waitKey(2000)

			# cv2.destroyWindow('Perspective Result Image')

			# cv2.destroyAllWindows()

			nonlocal rect_pts
			nonlocal hint
			if event == cv2.EVENT_LBUTTONUP:
				rect_pts.append([x, y])
				print("coordinate {} :{}".format(len(rect_pts), (x, y)))
				if len(rect_pts) == 4:
					print('映射變換')
					mappImg(rect_pts, copy.deepcopy(self.img))
					rect_pts = []
				else:
					print(hint[len(rect_pts)])

		cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
		cv2.imshow(window_name, self.img)
		cv2.setMouseCallback(window_name, select_points)


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
		self.btn1_3.clicked.connect(self.on_btn1_3_click)
		self.btn1_4.clicked.connect(self.on_btn1_4_click)
		self.btn2_1.clicked.connect(self.on_btn2_1_click)
		self.btn3_1.clicked.connect(self.on_btn3_1_click)
		self.btn4_1.clicked.connect(self.on_btn4_1_click)
		self.btn4_2.clicked.connect(self.on_btn4_2_click)
		self.btn5_1.clicked.connect(self.on_btn5_1_click)
		self.btn5_2.clicked.connect(self.on_btn5_2_click)

		# Init GUI
		self.edtAngle.setText('45')
		self.edtScale.setText('0.8')
		self.edtTx.setText('150')
		self.edtTy.setText('50')

	# button for problem 1.1
	def on_btn1_1_click(self):
		img = Img('images/dog.bmp')
		img.load()

	def on_btn1_2_click(self):
		img = Img('images/color.png')
		img.color_conver()

	def on_btn1_3_click(self):
		img = Img('images/dog.bmp')
		img.Flipping()

	def on_btn1_4_click(self):
		img = Img('images/dog.bmp')
		img.blending()

	def on_btn2_1_click(self):
		img = Img('images/M8.jpg')
		img.edge_detect()

	def on_btn3_1_click(self):
		img = Img('images/pyramids_Gray.jpg')
		img.Gauss_and_Laplac()

	def on_btn4_1_click(self):
		img = Img('images/QR.png')
		img.glo_hold()

	def on_btn4_2_click(self):
		img = Img('images/QR.png')
		img.loc_hold()

	def on_btn5_1_click(self):
		# edtAngle, edtScale. edtTx, edtTy to access to the ui object
		img = Img('images/OriginalTransform.png')
		img.show_Transform1(angle=int(self.edtAngle.displayText()),
							scale=float(self.edtScale.displayText()),
							tx=int(self.edtTx.displayText()),
							ty=int(self.edtTy.displayText()))

	def on_btn5_2_click(self):
		img = Img('images/OriginalPerspective.png')
		img.show_Transform2()

	### ### ###


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
