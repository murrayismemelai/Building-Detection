import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.filters.rank import entropy
from skimage.morphology import square
import matplotlib.pyplot as plt
import os
#from vegetation import vegetation_detection
from scipy.spatial import ConvexHull
import glob

class vegetation_detection:
    def vegetation_green(self,img):
        # chage data type for div
        img = img.astype(np.float64)
        green_channel = img[:,:,1]
        total_channel = img[:,:,0]+img[:,:,1]+img[:,:,2]

        #compute ref intensity
        green_intensity = np.divide(green_channel, total_channel, out=np.zeros_like(green_channel), where=total_channel != 0)

        #otsu generate binary image
        thresh = threshold_otsu(green_intensity)
        binary = (green_intensity > thresh)

        # green intensitu max value 1->255
        green_intensity = green_intensity * 255
        # output range 0~255
        return binary, green_intensity.astype(np.uint8)

class shadow_detection:
    def add_RGB_contraint(self, img):
        img = img.astype(np.float64)
        red = img[:, :, 2]
        green = img[:, :, 1]
        blue = img[:, :, 0]
        sum = red + green + blue
        mask = np.where(np.bitwise_and(sum < 200, np.var((red, green, blue), 0) < 80), 1, 0)
        return mask

    def shadow_new(self,img):
        # chage data type for div
        img = img.astype(np.float64)
        red_channel = img[:, :, 2]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 0]

        # add really black contraint
        black_mask = self.add_RGB_contraint(img)

        #compare red and all channek
        a = red_channel - np.sqrt((np.square(red_channel)+np.square(green_channel)+np.square(blue_channel)))
        b = red_channel + np.sqrt((np.square(red_channel) + np.square(green_channel) + np.square(blue_channel)))
        temp = np.divide(a, b, out=np.zeros_like(a),
                         where=b != 0)
        shadow_intensity = (4.0 / np.pi) * np.arctan(temp)

        # otsu thresholding
        thr = threshold_otsu(shadow_intensity)
        # shadow become white
        binary = (shadow_intensity < thr)

        # combine shadow and color constraint
        binary_w_mask = np.where(black_mask == 1, binary, False)
        return binary_w_mask, shadow_intensity

# optional function
class road_detection:
    def add_RGB_contraint(self,img):
        img = img.astype(np.float64)
        red = img[:, :, 2]
        green = img[:, :, 1]
        blue = img[:, :, 0]
        sum = red + green + blue
        mask = np.where(np.bitwise_and(np.bitwise_and(sum > 200, sum < 410), np.var((red, green, blue), 0) < 70), 1, 0)
        return mask

    def road_gb(self,img):
        #change data type for div
        img = img.astype(np.float64)
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 0]
        temp = np.divide((green_channel-blue_channel), (green_channel+blue_channel), out=np.zeros_like((green_channel-blue_channel)),
                                    where=(green_channel+blue_channel) != 0)

        #compute green and blue intensity
        blue_intensity = (4.0/np.pi)*np.arctan(temp)

        #otsu for binary image
        thresh = threshold_otsu(blue_intensity)
        binary = (blue_intensity < thresh)

        # add gray contraint
        road_mask = self.add_RGB_contraint(img)

        # masking gray part
        binary_w_mask = np.where(road_mask == 1, binary, False)
        '''
        #due to argtan, some negative value comes out, shift offset and mapping to 255
        if(np.min(green_intensity))<0:
            green_intensity = green_intensity + np.abs(np.min(green_intensity))
        green_intensity = (green_intensity/np.max(green_intensity))*255.0
        '''
        # output range 0~255
        return binary_w_mask, blue_intensity

# ground plane detection
class local_texture:
    def color_segmentation(self, img):
        # segment into 17 pieces 15*17 = 255
        color_seg = img / 15
        # modified 255 intensity to fit in 17 pieces
        img_seg = np.where(color_seg==17,16,color_seg)
        img_seg = img_seg * 15
        # offset to mid point
        return img_seg

    def entropy_filtering(self,img):
        color_seg = img
        #color_seg = self.color_segmentation(img)
        color_seg = cv2.cvtColor(color_seg, cv2.COLOR_BGR2GRAY)
        entr_img = entropy(color_seg,square(9))
        return entr_img

    def watershed_w_marker(self, img):
        entr_img = self.entropy_filtering(img)
        entr_thr = threshold_otsu(entr_img)

        # detect entropy
        entr_binary = (entr_img < entr_thr) * 255
        # entr_binary = (entr_img > np.max(entr_img)*0.87) * 255
        entr_binary = entr_binary.astype(np.uint8)

        '''
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # check sure foreground
        #ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        thr = threshold_otsu(dist_transform)
        sure_fg = (dist_transform>thr)*255
        #sure_fg = (dist_transform > np.max(dist_transform)*0.75) * 255
        sure_fg = np.uint8(sure_fg)
        '''

        # partition foreground
        ret, markers = cv2.connectedComponents(entr_binary)
        # ret, markers = cv2.connectedComponents(sure_fg)
        # markers = markers + 1

        # get watershed image , 0 is the seg line
        watershed_img = cv2.watershed(img, markers)
        watershed_img = watershed_img + 1
        return watershed_img

    def soft_veg_detection(self, img, veg_thr_ratio, large_area_num):
        vegetation_op = vegetation_detection()
        #blur_img = cv2.blur(img, (5, 5))
        blur_img = cv2.bilateralFilter(img, 5, 50,50)
        veg_binary, _ = vegetation_op.vegetation_green(blur_img)
        watershed_img = self.watershed_w_marker(img)

        # eliminate the area contain vegetation
        watershed_counts = np.bincount(np.ravel(watershed_img))
        veg_counts = np.bincount(np.ravel(np.where(veg_binary == True, watershed_img, 0)))
        if len(veg_counts) != len(watershed_counts):
            pad_length = len(watershed_counts) - len(veg_counts)
            zeros = np.zeros(pad_length, dtype=np.int)
            veg_counts = np.r_[veg_counts, zeros]

        watershed_counts = watershed_counts.astype(np.float)
        veg_counts = veg_counts.astype(np.float)
        veg_ratio = np.divide(veg_counts, watershed_counts, out=np.zeros_like(watershed_counts), where=veg_counts != 0)

        if veg_thr_ratio == 0:
            thr = threshold_otsu(veg_ratio[1:])
            index = np.where(veg_ratio[1:] > thr)[0] + 1
        else:
            index = np.where(veg_ratio[1:] > veg_thr_ratio)[0] + 1
        vegetation_mask = np.isin(watershed_img, index)

        # large area detection
        if large_area_num == 0:
            #area_thr = 100.0 * 100.0
            area_thr = threshold_otsu(watershed_counts[1:])
            index = np.where(watershed_counts[1:] > area_thr)[0] + 1
            large_area_mask = np.isin(watershed_img, index)
        else:
            candidate = np.argsort(-watershed_counts[0])[:large_area_num]
            large_area_mask = np.isin(watershed_img, candidate)

        return veg_binary, vegetation_mask, large_area_mask, watershed_img

    '''
    def large_area_detection(self,watershed_img,num_candidate):
        watershed_counts = np.bincount(np.ravel(watershed_img))
        if num_candidate == 0:
            thr = threshold_otsu(watershed_counts[1:])
            index = np.where(watershed_counts[1:] > thr)[0]
            large_area_mask = np.isin(watershed_img,index)
        else:
            candidate = np.argsort(-watershed_counts[0])[:num_candidate]
            large_area_mask = np.isin(watershed_img,candidate)
        return large_area_mask
    '''
    # too many computation
    '''
    def solidity_cal(self,water_shed,ratio):
        unique, counts = np.unique(water_shed, return_counts=True)
        stack = np.array([])
        for i in range(1,len(unique)):
            if counts[i]>10:
                print i
                print np.where(water_shed==unique[i])
                print '#################################'
                index_2d = np.where(water_shed==unique[i])
                hull = ConvexHull(zip(index_2d[0],index_2d[1]))
                if (hull.area/counts[i])<ratio:
                    stack = np.append(stack,unique[i])
        convex_mask = np.isin(water_shed,stack)
        return convex_mask
    '''

class groud_truth_compare:
    def total_detection(self,veg,soft_veg,area,shadow):
        temp = np.bitwise_or(veg, soft_veg)
        temp = np.bitwise_or(temp, area)
        #temp = np.bitwise_or(temp, road)
        temp = np.bitwise_or(temp, shadow)
        return temp

    def ref_gt(self,img,gt,bin_mask):
        overlap = np.copy(gt)
        ori_img = np.copy(img)
        gt_1_channel = gt[:,:,0]
        overlap[np.bitwise_and(bin_mask == True, gt_1_channel == 255) == True] = [0, 0, 255]
        #overlap[np.bitwise_and(bin_mask == False, gt_1_channel == 0) == True] = [255, 0, 0]
        ori_img[bin_mask == True] = [255, 255, 255]
        return ori_img, overlap

if __name__ == "__main__":
    # input testing file
    #test_file = 'AerialImageDataset/train/images/austin1.tif'
    #test_file = 'AerialImageDataset/train/images/austin6.tif'
    input_path = 'AerialImageDataset/train/images/austin27.tif'
    #input_file = 'tyrol-w24.tif'
    #input_file = 'chicago2.tif'
    #input_file = 'austin1.tif'
    test_path = 'AerialImageDataset/train/gt'

    for i, test_file in enumerate(glob.glob(input_path)):
        input_file = os.path.basename(test_file)
        print i, input_file
        #test_file = os.path.join(input_path, input_file)
        gt_file = os.path.join(test_path, input_file)

        # pre-process image
        test_img = cv2.imread(test_file)
        gt_img = cv2.imread(gt_file)
        dect_path = os.path.splitext(os.path.basename(input_file))[0]
        #dect_path = 'detection'

        if not os.path.exists(dect_path):
            os.makedirs(dect_path)

        # input class operation
        texture_op = local_texture()
        shadow_op = shadow_detection()
        road_op = road_detection()
        check_gt = groud_truth_compare()

        # ground plane detection
        vegetation_img, soft_veg_img, area_mask, watershed_markers = texture_op.soft_veg_detection(test_img,0.3,0)

        # shadow detection with black constraint
        shadow_img, _ = shadow_op.shadow_new(test_img)

        # road detection with gray constraint
        road_img, _ = road_op.road_gb(test_img)

        # combine all non-building mask
        #bin_detection = check_gt.total_detection(vegetation_img, soft_veg_img, area_mask, shadow_img, road_img)
        bin_detection = check_gt.total_detection(vegetation_img, soft_veg_img, area_mask, shadow_img)
        #bin_detection = check_gt.total_detection(vegetation_img, soft_veg_img, road_img, shadow_img)
        ref_img, overlap_img = check_gt.ref_gt(test_img, gt_img, bin_detection)

        # plot and save image
        cv2.imwrite(os.path.join(dect_path, input_file), test_img)
        #plt.imsave(os.path.join(dect_path, 'shadow.tif'), shadow_img, cmap='gray')
        #plt.imsave(os.path.join(dect_path, 'road.tif'), road_img, cmap='gray')
        #plt.imsave(os.path.join(dect_path, 'binary_vegetation.tif'), vegetation_img, cmap='gray')
        #plt.imsave(os.path.join(dect_path, 'watershed.tif'), watershed_markers, cmap='jet')
        #plt.imsave(os.path.join(dect_path, 'soft_vegetation.tif'), soft_veg_img, cmap='gray')
        #plt.imsave(os.path.join(dect_path, 'large_area.tif'), area_mask, cmap='gray')

        plt.imsave(os.path.join(dect_path, 'total_detection.tif'), bin_detection, cmap='gray')
        cv2.imwrite(os.path.join(dect_path, 'reference.tif'), ref_img)
        #cv2.imwrite(os.path.join(dect_path, 'overlap.tif'), overlap_img)

        # fig, ax = plt.subplots()
        # cax = plt.imshow(watershed_markers, cmap='gray')
        # cbar = fig.colorbar(cax)
        # plt.show()