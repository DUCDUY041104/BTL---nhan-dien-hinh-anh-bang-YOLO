import cv2
import numpy as np

# Đường dẫn đến các tệp cấu hình và trọng số của YOLOv3
classPath = "coco.names"
configPath = "yolov3.cfg"
weightPath = "yolov3.weights"

# Các tham số cho phát hiện đối tượng
scale = 0.00392  # Hệ số tỷ lệ để chuẩn hóa giá trị pixel (1/255.0)
conf_threshold = 0.5  # Ngưỡng độ tin cậy cho Non-Maximum Suppression (NMS)
nms_threshold = 0.4  # Ngưỡng IOU cho Non-Maximum Suppression (NMS)
CONFIDENCE = 0.5  # Ngưỡng độ tin cậy ban đầu để lọc các phát hiện


def settext(img, str_text, local):
    """
    Đặt một văn bản lên hình ảnh.

    Args:
        img (numpy.ndarray): Hình ảnh OpenCV.
        str_text (str): Chuỗi văn bản cần hiển thị.
        local (tuple): Tọa độ (x, y) của góc dưới bên trái của văn bản.
    """
    cv2.putText(img, str_text, local, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def getOutput(net):
    """
    Lấy tên của các lớp đầu ra không được kết nối của mạng nơ-ron.

    Args:
        net (cv2.dnn.Net): Đối tượng mạng nơ-ron đã được tải.

    Returns:
        list: Danh sách tên các lớp đầu ra.
    """
    layer_names = net.getLayerNames()
    # Lấy chỉ số của các lớp đầu ra không được kết nối
    outputLayerIndices = net.getUnconnectedOutLayers()
    # Chuyển đổi chỉ số thành tên lớp
    outputLayer = [layer_names[i - 1] for i in outputLayerIndices]
    return outputLayer


class LYL:
    """
    Lớp để thực hiện phát hiện đối tượng bằng cách sử dụng mô hình YOLOv3.
    """
    img = None
    boxes = []
    confidences = []
    classIds = []
    classes = []
    indices = None

    def init(self, im):
        """
        Khởi tạo và chạy quá trình phát hiện đối tượng trên một hình ảnh.

        Args:
            im (numpy.ndarray): Hình ảnh đầu vào để phát hiện đối tượng.

        Returns:
            LYL: Trả về đối tượng LYL đã được khởi tạo.
        """
        self.img = im
        # Đọc tên các lớp từ tệp coco.names
        with open(classPath, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        H, W, _ = self.img.shape  # Lấy chiều cao và chiều rộng của hình ảnh

        # Tải mô hình YOLOv3
        net = cv2.dnn.readNet(weightPath, configPath)

        # Tạo blob từ hình ảnh đầu vào
        # Blob là định dạng đầu vào cần thiết cho mạng nơ-ron
        blob = cv2.dnn.blobFromImage(self.img, scale, (416, 416), (0, 0, 0), True, crop=False)

        # Đặt blob làm đầu vào cho mạng
        net.setInput(blob)

        # Thực hiện truyền xuôi qua mạng để nhận được các phát hiện
        outs = net.forward(getOutput(net))

        # Xử lý các phát hiện từ mỗi lớp đầu ra
        for out in outs:
            for detection in out:
                scores = detection[5:]  # Lấy điểm số tin cậy cho từng lớp
                classId = np.argmax(scores)  # Lấy chỉ số của lớp có điểm số cao nhất
                confidence = scores[classId]  # Lấy độ tin cậy của lớp được phát hiện

                # Chỉ giữ lại các phát hiện có độ tin cậy cao hơn ngưỡng CONFIDENCE
                if confidence > CONFIDENCE:
                    # Tính toán tọa độ trung tâm, chiều rộng và chiều cao của hộp giới hạn
                    centerX = int(detection[0] * W)
                    centerY = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)

                    # Tính toán tọa độ góc trên bên trái của hộp giới hạn
                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)

                    # Thêm thông tin phát hiện vào các danh sách
                    self.classIds.append(classId)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])

        # Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các hộp giới hạn trùng lặp
        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, conf_threshold, nms_threshold)
        return self

    def countObj(self, obj):
        """
        Đếm số lần xuất hiện của một đối tượng cụ thể trong các phát hiện.

        Args:
            obj (str): Tên của đối tượng cần đếm (ví dụ: "person").

        Returns:
            int: Số lượng đối tượng được đếm.
        """
        count = 0
        # Vòng lặp qua các chỉ số của các hộp giới hạn đã được NMS giữ lại
        if len(self.indices) > 0:
            for i in self.indices.flatten():  # .flatten() để xử lý trường hợp chỉ có 1 phát hiện
                label = str(self.classes[self.classIds[i]])
                if label == obj:
                    count = count + 1
        return count

    def drawObj(self, obj, withLabel=False):
        """
        Vẽ các hộp giới hạn trên hình ảnh cho một đối tượng cụ thể hoặc cho tất cả các đối tượng.

        Args:
            obj (str): Tên của đối tượng cần vẽ hộp giới hạn. Nếu là chuỗi rỗng (""), nó sẽ vẽ tất cả các đối tượng.
            withLabel (bool, optional): Nếu True, sẽ vẽ tên lớp bên cạnh hộp giới hạn. Mặc định là False.

        Returns:
            numpy.ndarray: Hình ảnh đã được vẽ các hộp giới hạn.
        """
        img = self.img
        # Vòng lặp qua các chỉ số của các hộp giới hạn đã được NMS giữ lại
        if len(self.indices) > 0:
            for i in self.indices.flatten():  # .flatten() để xử lý trường hợp chỉ có 1 phát hiện
                box = self.boxes[i]
                x1 = box[0]
                y1 = box[1]
                w1 = box[2]
                h1 = box[3]
                label = str(self.classes[self.classIds[i]])

                # Kiểm tra xem đối tượng có khớp với obj hoặc nếu obj là chuỗi rỗng (vẽ tất cả)
                if label == obj or obj == "":
                    # Vẽ hình chữ nhật màu xanh lá cây xung quanh đối tượng
                    cv2.rectangle(img, (round(x1), round(y1)), (round(x1 + w1), round(y1 + h1)), (0, 255, 0), 2)
                    if withLabel == True:
                        # Vẽ tên lớp nếu withLabel là True
                        cv2.putText(img, label, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return img


if __name__ == "__main__":
    # Ví dụ sử dụng:
    # Đảm bảo bạn có các tệp 'coco.names', 'yolov3.cfg', 'yolov3.weights' trong cùng thư mục
    # và một hình ảnh để kiểm tra, ví dụ: 'test_image.jpg'

    # Tải một hình ảnh
    image_path = "test_image.jpg"  # THAY ĐỔI ĐƯỜNG DẪN NÀY ĐẾN HÌNH ẢNH CỦA BẠN
    img = cv2.imread(image_path)

    if img is None:
        print(f"Lỗi: Không thể tải hình ảnh từ: {image_path}")
        print("Vui lòng đảm bảo đường dẫn hình ảnh là chính xác và tệp tồn tại.")
    else:
        print("Đang khởi tạo bộ phát hiện đối tượng...")
        detector = LYL().init(img)
        print("Phát hiện đối tượng hoàn tất.")

        # Đếm số lượng người (hoặc bất kỳ đối tượng nào khác có trong coco.names)
        person_count = detector.countObj("person")
        print(f"Số lượng 'person' được phát hiện: {person_count}")

        # Đếm số lượng xe hơi
        car_count = detector.countObj("car")
        print(f"Số lượng 'car' được phát hiện: {car_count}")

        # Vẽ hộp giới hạn cho tất cả các đối tượng với nhãn
        print("Đang vẽ hộp giới hạn lên hình ảnh...")
        img_with_boxes = detector.drawObj("", withLabel=True)  # "" để vẽ tất cả các đối tượng

        # Hiển thị hình ảnh kết quả
        cv2.imshow("Kết quả phát hiện đối tượng YOLOv3", img_with_boxes)
        print("Nhấn phím bất kỳ để đóng cửa sổ.")
        cv2.waitKey(0)  # Chờ phím bấm bất kỳ
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV
