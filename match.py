python

class ImageStitcher:
    def __init__(self, src, des):
        self.src = src
        self.des = des
        self.GOOD_POINTS_LIMITED = 0.99

    def stitch_images(self):
        img1_3 = cv.imread(self.src, 1)  # 基准图像
        img2_3 = cv.imread(self.des, 1)  # 拼接图像

        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1_3, None)
        kp2, des2 = orb.detectAndCompute(img2_3, None)

        bf = cv.BFMatcher.create()

        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        goodPoints = []
        for i in range(len(matches) - 1):
            if matches[i].distance < self.GOOD_POINTS_LIMITED * matches[i + 1].distance:
                goodPoints.append(matches[i])

        src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(dst_pts, src_pts, cv.RHO)

        h1, w1, p1 = img2_3.shape
        h2, w2, p2 = img1_3.shape

        h = np.maximum(h1, h2)
        w = np.maximum(w1, w2)
        ...
        ...
        dst = cv.add(dst1, imageTransform)
        dst_no = np.copy(dst)

        dst_target = np.maximum(dst1, imageTransform)

        return dst_target
