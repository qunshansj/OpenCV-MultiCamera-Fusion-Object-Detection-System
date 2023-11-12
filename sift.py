python

class Stitcher:
    def __init__(self):
        global model
        model = ['ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE']
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        (imageB, imageA) = images
        start = time.time()
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        end = time.time()
        print('%.5f s' % (end - start))

        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        start = time.time()
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)
        end = time.time()
        print('%.5f s' % (end - start))

        if M is None:
            return None



        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)
            end = time.time()
            print('%.5f s' % (end - start))
            return (result, vis)

        return result

    def detectAndDescribe(self, image):
        global model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        else:
            detector = cv2.FeatureDetector_create(model[0])
            kps = detector.detect(gray)

            extractor = cv2.DescriptorExtractor_create(model[0])
            (kps, features) = extractor.compute(gray, kps)

        kps = np.float32([kp.pt for kp in kps])

        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        ...
        ...
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis


